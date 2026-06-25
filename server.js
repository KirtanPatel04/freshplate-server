import 'dotenv/config';
import express from 'express';
import rateLimit from 'express-rate-limit';
import { createClient } from '@supabase/supabase-js';
import { WebSocket } from 'ws';

// Node.js <21 doesn't have a global WebSocket — Supabase realtime requires it
if (!globalThis.WebSocket) globalThis.WebSocket = WebSocket;
import { readFileSync, writeFileSync } from 'fs';

const app  = express();
const PORT = process.env.PORT || 3000;

// Prevent Railway from restarting the server on unhandled rejections
process.on('uncaughtException',   err    => console.error('[uncaughtException]', err));
process.on('unhandledRejection',  reason => console.error('[unhandledRejection]', reason));

// ---------- Middleware ----------

app.set('trust proxy', 1);
app.use(express.json({ limit: '25mb' }));

// ── Per-user rate limiting ─────────────────────────────────────────────────────
// Use a fast hash of the Authorization token as the rate-limit key so that every
// authenticated user gets their own independent bucket regardless of shared IPs
// (NAT, corporate proxy, etc.).  Falls back to IP for unauthenticated requests.
function userKey(req) {
  const auth = req.headers['authorization'];
  if (auth?.startsWith('Bearer ')) {
    const token = auth.slice(7, 57); // first 50 chars is plenty for a unique hash
    let h = 5381;
    for (let i = 0; i < token.length; i++) h = ((h << 5) + h + token.charCodeAt(i)) | 0;
    return 'u:' + (h >>> 0).toString(36);
  }
  return 'ip:' + (req.ip ?? 'unknown');
}

// ── Rate limiters ──────────────────────────────────────────────────────────────
// Tier 1 — per-minute burst guard (all AI endpoints)
const perMinuteLimiter = rateLimit({
  windowMs: 60_000,
  max: 30,
  keyGenerator: userKey,
  standardHeaders: true,
  legacyHeaders: false,
  validate: { xForwardedForHeader: false },
  message: { error: 'Too many requests. Please slow down.' }
});

// Tier 2 — per-day overall AI budget (200 calls/user/day)
const dailyLimiter = rateLimit({
  windowMs: 24 * 60 * 60 * 1000,
  max: 200,
  keyGenerator: userKey,
  standardHeaders: true,
  legacyHeaders: false,
  validate: { xForwardedForHeader: false },
  message: { error: 'Daily limit reached. Come back tomorrow!' }
});

// Tier 3 — strict daily limits on expensive endpoints
const mealPlanDailyLimiter = rateLimit({
  windowMs: 24 * 60 * 60 * 1000,
  max: 5,
  keyGenerator: userKey,
  standardHeaders: true,
  legacyHeaders: false,
  validate: { xForwardedForHeader: false },
  message: { error: 'Meal plan limit: 5 per day. Come back tomorrow!' }
});

const recipeDailyLimiter = rateLimit({
  windowMs: 24 * 60 * 60 * 1000,
  max: 30,
  keyGenerator: userKey,
  standardHeaders: true,
  legacyHeaders: false,
  validate: { xForwardedForHeader: false },
  message: { error: 'Recipe search limit reached for today.' }
});

const notifDailyLimiter = rateLimit({
  windowMs: 24 * 60 * 60 * 1000,
  max: 10,
  keyGenerator: userKey,
  standardHeaders: true,
  legacyHeaders: false,
  validate: { xForwardedForHeader: false },
  message: { error: 'Notification generation limit reached for today.' }
});


// Stack the two tiers: every AI route uses perMinute + daily
const aiLimiter = [perMinuteLimiter, dailyLimiter];

// ---------- Gemini helpers ----------

const GEMINI_BASE = 'https://generativelanguage.googleapis.com/v1beta/models';
// Waterfall: try each model in order; move to next on quota/overload errors.
const MODEL_WATERFALL = [
  'gemini-2.5-flash-lite',  // primary — cheapest, fastest
  'gemini-2.5-flash',       // fallback — same generation, higher output limits
  'gemini-3.1-flash-lite',  // last resort — newer generation, higher quality
];
const MODEL_MAIN  = MODEL_WATERFALL[0];
const MODEL_CHECK = MODEL_WATERFALL[0];

// ── API key rotation ───────────────────────────────────────────────────────────
// Set GEMINI_API_KEYS=key1,key2,key3 in .env to spread load across multiple keys
// and multiply your effective quota (e.g. 3 keys × 10 RPM = 30 RPM effective).
// Falls back to the single GEMINI_API_KEY for backward compat.
const GEMINI_KEYS = (process.env.GEMINI_API_KEYS ?? process.env.GEMINI_API_KEY ?? '')
  .split(',').map(k => k.trim()).filter(Boolean);
let _keyIdx = 0;
function pickApiKey() {
  if (!GEMINI_KEYS.length) throw new Error('GEMINI_API_KEY is not set on the server.');
  const key = GEMINI_KEYS[_keyIdx % GEMINI_KEYS.length];
  _keyIdx = (_keyIdx + 1) % GEMINI_KEYS.length;
  return key;
}

// ── Concurrency limiter ────────────────────────────────────────────────────────
// Prevents a burst of users from flooding the Gemini API simultaneously.
// Excess requests queue here and are served in order as slots free up.
// Tune GEMINI_MAX_CONCURRENT ≈ (total RPM across all keys / avg seconds per call).
class Semaphore {
  #max; #active = 0; #queue = [];
  constructor(max) { this.#max = max; }
  acquire() {
    if (this.#active < this.#max) { this.#active++; return Promise.resolve(); }
    return new Promise(r => this.#queue.push(r)).then(() => { this.#active++; });
  }
  release() { this.#active--; if (this.#queue.length) this.#queue.shift()(); }
}
const geminiSem = new Semaphore(Number(process.env.GEMINI_MAX_CONCURRENT) || 20);

// ── Recipe search cache ────────────────────────────────────────────────────────
// If two users search for the same dish, only one Gemini call is made.
// Results are reused for 1 hour, then evicted so content stays fresh.
const _recipeCache = new Map();
const RECIPE_TTL = 60 * 60 * 1000; // 1 hour

function getCachedRecipes(key) {
  const e = _recipeCache.get(key);
  if (!e || Date.now() > e.exp) { _recipeCache.delete(key); return null; }
  return e.data;
}
function setCachedRecipes(key, data) {
  if (_recipeCache.size >= 500) _recipeCache.delete(_recipeCache.keys().next().value);
  _recipeCache.set(key, { data, exp: Date.now() + RECIPE_TTL });
}

async function callGemini(parts, { maxTokens = 8000, temperature = 0.7, model = MODEL_MAIN, models = null } = {}) {
  const apiKey = pickApiKey();

  const body = JSON.stringify({
    contents: [{ role: 'user', parts }],
    generationConfig: { maxOutputTokens: maxTokens, temperature }
  });

  // `models` overrides the full cascade — use it to pin large requests to flash only.
  // Otherwise build the standard cascade starting from `model`.
  const cascade = models ?? [model, ...MODEL_WATERFALL.filter(m => m !== model)];

  const MAX_ATTEMPTS = 3;
  let lastError;

  await geminiSem.acquire();
  try {
    for (const currentModel of cascade) {
      for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
        const res = await fetch(`${GEMINI_BASE}/${currentModel}:generateContent?key=${apiKey}`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' }, body
        });

        if ((res.status === 503 || res.status === 429) && attempt < MAX_ATTEMPTS) {
          await res.text(); // drain body so the connection is released back to the pool
          await new Promise(r => setTimeout(r, attempt * 2000));
          continue;
        }

        if (res.status === 503 || res.status === 429) {
          await res.text(); // drain body so the connection is released back to the pool
          lastError = new Error(`Gemini API ${res.status} on ${currentModel}`);
          break;
        }

        // 400 usually means this model doesn't support the request (e.g. no vision) — fall through
        if (res.status === 400) {
          const text = await res.text();
          lastError = new Error(`Gemini API 400 on ${currentModel}`);
          console.warn(`[callGemini] ${currentModel} returned 400, trying next model`);
          break;
        }

        if (!res.ok) {
          const text = await res.text();
          throw new Error(`Gemini API ${res.status}: ${text}`);
        }

        const data = await res.json();
        return data.candidates?.[0]?.content?.parts?.[0]?.text ?? '';
      }
    }
  } finally {
    geminiSem.release();
  }

  throw lastError ?? new Error('All Gemini models unavailable');
}

// Extract first JSON object or array from text (tolerates model preamble)
function extractJSON(text) {
  const arrIdx = text.indexOf('[');
  const objIdx = text.indexOf('{');

  let start, openChar, closeChar;
  if (arrIdx !== -1 && (objIdx === -1 || arrIdx < objIdx)) {
    start = arrIdx; openChar = '['; closeChar = ']';
  } else if (objIdx !== -1) {
    start = objIdx; openChar = '{'; closeChar = '}';
  } else {
    return text;
  }

  // Walk the string with proper bracket + string tracking so a ] inside
  // a step like "Season [to taste]" never terminates the extraction early.
  let depth = 0, inStr = false, esc = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (esc)              { esc = false; continue; }
    if (ch === '\\' && inStr) { esc = true;  continue; }
    if (ch === '"')       { inStr = !inStr;  continue; }
    if (inStr)            continue;
    if (ch === openChar)  depth++;
    else if (ch === closeChar) { if (--depth === 0) return text.slice(start, i + 1); }
  }
  return text.slice(start); // truncated — return what we have
}

function sanitize(str, maxLen = 300) {
  if (typeof str !== 'string') return '';
  return str.replace(/[<>'"]/g, '').slice(0, maxLen).trim();
}

// ---------- Supabase client ----------

const supabase = (process.env.SUPABASE_URL && process.env.SUPABASE_SERVICE_ROLE_KEY)
  ? createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY, {
      auth: { autoRefreshToken: false, persistSession: false }
    })
  : null;

function requireSupabase(req, res, next) {
  if (!supabase) return res.status(503).json({ error: 'Auth service not configured on the server.' });
  next();
}

// ---------- Auth routes ----------

// POST /auth/signup
app.post('/auth/signup', requireSupabase, async (req, res) => {
  try {
    const { email, password } = req.body ?? {};
    if (!email || !password) return res.status(400).json({ error: 'Email and password required.' });

    const { data: created, error: createErr } = await supabase.auth.admin.createUser({
      email: email.toLowerCase().trim(),
      password,
      email_confirm: true
    });
    if (createErr) {
      const msg = createErr.message ?? '';
      if (msg.toLowerCase().includes('already') || msg.toLowerCase().includes('exists')) {
        return res.status(400).json({ error: 'An account with this email already exists.' });
      }
      return res.status(400).json({ error: msg || 'Could not create account.' });
    }

    const { data: session, error: signInErr } = await supabase.auth.signInWithPassword({
      email: email.toLowerCase().trim(), password
    });
    if (signInErr) return res.status(400).json({ error: signInErr.message });

    res.json({
      access_token:  session.session.access_token,
      refresh_token: session.session.refresh_token,
      user_id:       session.user.id,
      email:         session.user.email
    });
  } catch (err) {
    console.error('[auth/signup]', err);
    res.status(500).json({ error: 'Server error during sign up.' });
  }
});

// POST /auth/signin
app.post('/auth/signin', requireSupabase, async (req, res) => {
  try {
    const { email, password } = req.body ?? {};
    if (!email || !password) return res.status(400).json({ error: 'Email and password required.' });

    const { data, error } = await supabase.auth.signInWithPassword({
      email: email.toLowerCase().trim(), password
    });
    if (error) return res.status(401).json({ error: 'Invalid email or password.' });

    res.json({
      access_token:  data.session.access_token,
      refresh_token: data.session.refresh_token,
      user_id:       data.user.id,
      email:         data.user.email
    });
  } catch (err) {
    console.error('[auth/signin]', err);
    res.status(500).json({ error: 'Server error during sign in.' });
  }
});

// POST /auth/google  — iOS sends the Google ID token from its PKCE flow
app.post('/auth/google', requireSupabase, async (req, res) => {
  try {
    const { id_token } = req.body ?? {};
    if (!id_token) return res.status(400).json({ error: 'Google ID token required.' });

    const { data, error } = await supabase.auth.signInWithIdToken({
      provider: 'google',
      token: id_token
    });
    if (error) return res.status(400).json({ error: error.message });

    const meta = data.user.user_metadata ?? {};
    res.json({
      access_token:  data.session.access_token,
      refresh_token: data.session.refresh_token,
      user_id:       data.user.id,
      email:         data.user.email ?? '',
      name:          meta.full_name ?? meta.name ?? ''
    });
  } catch (err) {
    console.error('[auth/google]', err);
    res.status(500).json({ error: 'Server error during Google sign in.' });
  }
});

// POST /auth/apple  — iOS sends the Apple identity token from Sign in with Apple
app.post('/auth/apple', requireSupabase, async (req, res) => {
  try {
    const { identity_token } = req.body ?? {};
    if (!identity_token) return res.status(400).json({ error: 'Apple identity token required.' });

    const { data, error } = await supabase.auth.signInWithIdToken({
      provider: 'apple',
      token: identity_token
    });
    if (error) return res.status(400).json({ error: error.message });

    const meta = data.user.user_metadata ?? {};
    res.json({
      access_token:  data.session.access_token,
      refresh_token: data.session.refresh_token,
      user_id:       data.user.id,
      email:         data.user.email ?? '',
      name:          meta.full_name ?? meta.name ?? ''
    });
  } catch (err) {
    console.error('[auth/apple]', err);
    res.status(500).json({ error: 'Server error during Apple sign in.' });
  }
});

// POST /auth/refresh
app.post('/auth/refresh', requireSupabase, async (req, res) => {
  try {
    const { refresh_token } = req.body ?? {};
    if (!refresh_token) return res.status(400).json({ error: 'Refresh token required.' });

    const { data, error } = await supabase.auth.refreshSession({ refresh_token });
    if (error) return res.status(401).json({ error: 'Session expired. Please sign in again.' });

    res.json({
      access_token:  data.session.access_token,
      refresh_token: data.session.refresh_token
    });
  } catch (err) {
    console.error('[auth/refresh]', err);
    res.status(500).json({ error: 'Server error during token refresh.' });
  }
});

// POST /auth/send-reset-code  — sends a 6-digit OTP to the user's email
app.post('/auth/send-reset-code', requireSupabase, async (req, res) => {
  try {
    const { email } = req.body ?? {};
    if (!email) return res.status(400).json({ error: 'Email required.' });

    await supabase.auth.signInWithOtp({
      email: email.toLowerCase().trim(),
      options: { shouldCreateUser: true }
    });
    res.json({ message: 'If an account exists, a 6-digit code has been sent.' });
  } catch (err) {
    console.error('[auth/send-reset-code]', err);
    res.status(500).json({ error: 'Server error. Please try again.' });
  }
});

// POST /auth/verify-reset-code  — verifies the 6-digit code and returns an access token
app.post('/auth/verify-reset-code', requireSupabase, async (req, res) => {
  try {
    const { email, code } = req.body ?? {};
    if (!email || !code) return res.status(400).json({ error: 'Email and code required.' });

    const { data, error } = await supabase.auth.verifyOtp({
      email: email.toLowerCase().trim(),
      token: code,
      type: 'email'
    });
    if (error || !data?.session) return res.status(400).json({ error: 'Incorrect or expired code. Please try again.' });

    res.json({ access_token: data.session.access_token });
  } catch (err) {
    console.error('[auth/verify-reset-code]', err);
    res.status(500).json({ error: 'Server error. Please try again.' });
  }
});

// POST /auth/update-password  — called after user taps reset link in email
app.post('/auth/update-password', requireSupabase, async (req, res) => {
  try {
    const { access_token, new_password } = req.body ?? {};
    if (!access_token || !new_password) return res.status(400).json({ error: 'Token and password required.' });
    if (new_password.length < 6) return res.status(400).json({ error: 'Password must be at least 6 characters.' });

    // Verify the recovery token and get the user
    const { data: { user }, error: userError } = await supabase.auth.getUser(access_token);
    if (userError || !user) return res.status(401).json({ error: 'Invalid or expired reset link. Please request a new one.' });

    // Update the password
    const { error } = await supabase.auth.admin.updateUserById(user.id, { password: new_password });
    if (error) return res.status(400).json({ error: error.message });

    res.json({ message: 'Password updated successfully.' });
  } catch (err) {
    console.error('[auth/update-password]', err);
    res.status(500).json({ error: 'Server error. Please try again.' });
  }
});

// ---------- Health ----------

app.get('/health', (_req, res) => {
  console.log('[health] ping');
  res.json({ ok: true });
});

// ---------- User Profile ----------

async function getAuthUser(req, res) {
  const authHeader = req.headers.authorization || '';
  const token = authHeader.startsWith('Bearer ') ? authHeader.slice(7) : req.body?.access_token;
  if (!token) { res.status(401).json({ error: 'Missing auth token.' }); return null; }
  const { data: { user }, error } = await supabase.auth.getUser(token);
  if (error || !user) { res.status(401).json({ error: 'Unauthorized.' }); return null; }
  return user;
}

function extractDisplayName(supabaseUser, fallback) {
  const meta = supabaseUser?.user_metadata ?? {};
  return meta.display_name || meta.full_name || meta.name || fallback || supabaseUser?.email?.split('@')[0] || 'FreshPlate User';
}

// POST /users/sync-profile — store display name for friend lookup
app.post('/users/sync-profile', requireSupabase, async (req, res) => {
  try {
    const user = await getAuthUser(req, res);
    if (!user) return;
    const { display_name } = req.body ?? {};
    if (!display_name) return res.status(400).json({ error: 'display_name required.' });

    await supabase.from('user_profiles').upsert(
      { id: user.id, email: user.email, display_name },
      { onConflict: 'id' }
    );
    res.json({ success: true });
  } catch (err) {
    console.error('[users/sync-profile]', err);
    res.status(500).json({ error: 'Server error.' });
  }
});

// ---------- Friends ----------

// POST /friends/lookup — check if an email has a FreshPlate account
app.post('/friends/lookup', requireSupabase, async (req, res) => {
  try {
    const user = await getAuthUser(req, res);
    if (!user) return;

    const { email } = req.body ?? {};
    if (!email) return res.status(400).json({ error: 'Email required.' });
    const normalised = email.toLowerCase().trim();

    if (normalised === user.email?.toLowerCase()) {
      return res.status(400).json({ error: "That's your own account — try a friend's email!" });
    }

    // Look up in user_profiles table first (populated via sync-profile on login)
    const { data: profile } = await supabase
      .from('user_profiles')
      .select('id, email, display_name')
      .eq('email', normalised)
      .maybeSingle();

    if (!profile) {
      return res.status(404).json({ error: 'No FreshPlate account found with that email.' });
    }

    // Check for existing relationship
    const { data: existing } = await supabase
      .from('friend_requests')
      .select('status')
      .or(`and(sender_id.eq.${user.id},receiver_id.eq.${profile.id}),and(sender_id.eq.${profile.id},receiver_id.eq.${user.id})`)
      .maybeSingle();

    return res.json({
      userId: profile.id,
      email: profile.email,
      name: profile.display_name || normalised.split('@')[0],
      existingStatus: existing?.status || null
    });
  } catch (err) {
    console.error('[friends/lookup]', err);
    res.status(500).json({ error: 'Server error.' });
  }
});

// POST /friends/request — send a friend request
app.post('/friends/request', requireSupabase, async (req, res) => {
  try {
    const user = await getAuthUser(req, res);
    if (!user) return;

    const { receiver_id, sender_name, receiver_name } = req.body ?? {};
    if (!receiver_id) return res.status(400).json({ error: 'receiver_id required.' });
    if (user.id === receiver_id) return res.status(400).json({ error: "You can't add yourself." });

    // Block if already accepted
    const { data: existing } = await supabase
      .from('friend_requests')
      .select('status')
      .or(`and(sender_id.eq.${user.id},receiver_id.eq.${receiver_id}),and(sender_id.eq.${receiver_id},receiver_id.eq.${user.id})`)
      .maybeSingle();

    if (existing?.status === 'accepted') {
      return res.status(400).json({ error: 'Already friends.' });
    }

    const { error } = await supabase.from('friend_requests').upsert({
      sender_id: user.id,
      receiver_id,
      sender_name: sender_name || extractDisplayName(user, ''),
      receiver_name: receiver_name || '',
      status: 'pending'
    }, { onConflict: 'sender_id,receiver_id' });

    if (error) throw error;
    res.json({ success: true });
  } catch (err) {
    console.error('[friends/request]', err);
    res.status(500).json({ error: 'Server error.' });
  }
});

// GET /friends/incoming — pending requests addressed to me
app.get('/friends/incoming', requireSupabase, async (req, res) => {
  try {
    const user = await getAuthUser(req, res);
    if (!user) return;

    const { data, error } = await supabase
      .from('friend_requests')
      .select('*')
      .eq('receiver_id', user.id)
      .eq('status', 'pending')
      .order('created_at', { ascending: false });

    if (error) throw error;

    const requests = await Promise.all((data || []).map(async row => {
      let senderName = row.sender_name || '';
      let senderEmail = '';
      try {
        const { data: profile } = await supabase
          .from('user_profiles')
          .select('email, display_name')
          .eq('id', row.sender_id)
          .maybeSingle();
        if (profile) { senderName = profile.display_name || senderName; senderEmail = profile.email; }
      } catch {}
      return { id: row.id, senderUserId: row.sender_id, senderName, senderEmail, createdAt: row.created_at };
    }));

    res.json({ requests });
  } catch (err) {
    console.error('[friends/incoming]', err);
    res.status(500).json({ error: 'Server error.' });
  }
});

// GET /friends/outgoing — my pending outgoing requests
app.get('/friends/outgoing', requireSupabase, async (req, res) => {
  try {
    const user = await getAuthUser(req, res);
    if (!user) return;

    const { data, error } = await supabase
      .from('friend_requests')
      .select('*')
      .eq('sender_id', user.id)
      .eq('status', 'pending')
      .order('created_at', { ascending: false });

    if (error) throw error;

    const requests = await Promise.all((data || []).map(async row => {
      let receiverName = row.receiver_name || '';
      let receiverEmail = '';
      try {
        const { data: profile } = await supabase
          .from('user_profiles')
          .select('email, display_name')
          .eq('id', row.receiver_id)
          .maybeSingle();
        if (profile) { receiverName = profile.display_name || receiverName; receiverEmail = profile.email; }
      } catch {}
      return { id: row.id, receiverUserId: row.receiver_id, receiverName, receiverEmail, createdAt: row.created_at };
    }));

    res.json({ requests });
  } catch (err) {
    console.error('[friends/outgoing]', err);
    res.status(500).json({ error: 'Server error.' });
  }
});

// POST /friends/accept — accept an incoming request
app.post('/friends/accept', requireSupabase, async (req, res) => {
  try {
    const user = await getAuthUser(req, res);
    if (!user) return;

    const { request_id } = req.body ?? {};
    if (!request_id) return res.status(400).json({ error: 'request_id required.' });

    const myName = extractDisplayName(user, '');
    const { data, error } = await supabase
      .from('friend_requests')
      .update({ status: 'accepted', receiver_name: myName })
      .eq('id', request_id)
      .eq('receiver_id', user.id)
      .select()
      .single();

    if (error || !data) return res.status(404).json({ error: 'Request not found.' });
    res.json({ success: true });
  } catch (err) {
    console.error('[friends/accept]', err);
    res.status(500).json({ error: 'Server error.' });
  }
});

// POST /friends/decline — decline or cancel a request
app.post('/friends/decline', requireSupabase, async (req, res) => {
  try {
    const user = await getAuthUser(req, res);
    if (!user) return;

    const { request_id } = req.body ?? {};
    if (!request_id) return res.status(400).json({ error: 'request_id required.' });

    const { error } = await supabase
      .from('friend_requests')
      .delete()
      .eq('id', request_id)
      .or(`receiver_id.eq.${user.id},sender_id.eq.${user.id}`);

    if (error) throw error;
    res.json({ success: true });
  } catch (err) {
    console.error('[friends/decline]', err);
    res.status(500).json({ error: 'Server error.' });
  }
});

// ---------- Scan Food ----------

app.post('/api/ai/scan-food', ...aiLimiter, async (req, res) => {
  const { image, mimeType = 'image/jpeg' } = req.body;
  console.log('[scan-food] request received');
  if (!image) return res.status(400).json({ error: 'image is required' });

  try {
    const text = await callGemini(
      [
        { inlineData: { mimeType, data: image } },
        {
          text: 'Identify the food in this image and estimate nutritional info for a typical single serving. ' +
                'Return ONLY compact JSON with no extra text or markdown: ' +
                '{"name":"str","emoji":"str","calories":0,"protein":0.0,"carbs":0.0,"fat":0.0,"fiber":0.0,"servingSize":"str"}'
        }
      ],
      { maxTokens: 300, temperature: 0.2 }
    );
    res.json(JSON.parse(extractJSON(text)));
  } catch (err) {
    console.error('[scan-food]', err.message);
    res.status(500).json({ error: err.message });
  }
});

// ---------- Analyze Meal Photo ----------

app.post('/api/ai/analyze-meal', ...aiLimiter, async (req, res) => {
  const { image, mimeType = 'image/jpeg' } = req.body;
  console.log('[analyze-meal] request received');
  if (!image) return res.status(400).json({ error: 'image is required' });

  try {
    const text = await callGemini(
      [
        { inlineData: { mimeType, data: image } },
        {
          text: 'Analyze this meal photo in detail. Identify every ingredient visible and estimate the full nutritional breakdown for the entire plate. ' +
                'Return ONLY compact JSON: ' +
                '{"name":"str","emoji":"str","calories":0,"protein":0.0,"carbs":0.0,"fat":0.0,"fiber":0.0,"servingSize":"str","ingredients":["ingredient with qty"]}'
        }
      ],
      { maxTokens: 500, temperature: 0.2 }
    );
    res.json(JSON.parse(extractJSON(text)));
  } catch (err) {
    console.error('[analyze-meal]', err.message);
    res.status(500).json({ error: err.message });
  }
});

// ---------- Analyze Water ----------

app.post('/api/ai/analyze-water', ...aiLimiter, async (req, res) => {
  const { image, mimeType = 'image/jpeg', unit = 'mL' } = req.body;
  console.log('[analyze-water] request received');
  if (!image) return res.status(400).json({ error: 'image is required' });

  const safeUnit = sanitize(unit, 20);
  try {
    const text = await callGemini(
      [
        { inlineData: { mimeType, data: image } },
        {
          text: `Estimate the amount of water or liquid in this image. Return ONLY: {"amount": <number>}\n` +
                `Unit: ${safeUnit}. A standard glass ≈ 250 mL / 8 fl oz. Return 0 if no liquid visible.`
        }
      ],
      { maxTokens: 100, temperature: 0.1 }
    );
    const json = JSON.parse(extractJSON(text));
    res.json({ amount: Number(json.amount) || 0 });
  } catch (err) {
    console.error('[analyze-water]', err.message);
    res.status(500).json({ error: err.message });
  }
});

// ---------- Search Recipes (with food safety check) ----------

// Quick local blocklist — saves an API call for obvious non-food queries
const NON_FOOD_TERMS = new Set([
  'porn','sex','nude','naked','drugs','cocaine','heroin','meth','weed','marijuana',
  'hack','virus','malware','exploit','weapon','gun','bomb','kill','murder','suicide',
  'politics','election','president','war','crypto','bitcoin','stock','investment',
  'login','password','credit card','social security'
]);

function isObviouslyNotFood(query) {
  const lower = query.toLowerCase();
  return [...NON_FOOD_TERMS].some(term => lower.includes(term));
}

app.post('/api/ai/search-recipes', ...aiLimiter, recipeDailyLimiter, async (req, res) => {
  const { query, pantryItems = [] } = req.body;
  console.log('[search-recipes] query:', query);
  if (!query) return res.status(400).json({ error: 'query is required' });

  const safeQuery = sanitize(query, 200);

  // Stage 1 — fast local blocklist (zero API cost)
  if (isObviouslyNotFood(safeQuery)) {
    return res.status(400).json({
      error: 'not_food',
      message: 'Please search for a food or recipe — FreshPlate is your kitchen companion! 🍽️'
    });
  }

  // Stage 1b — in-memory cache hit (skip Gemini entirely if we've seen this query recently)
  // Cache key ignores pantry items so the base recipes are shared across users.
  const cacheKey = safeQuery.toLowerCase().trim();
  const cached = getCachedRecipes(cacheKey);
  if (cached) {
    console.log('[search-recipes] cache hit:', safeQuery);
    return res.json(cached);
  }

  // Stage 2 — simple YES/NO food check (cheap: ~5 tokens out)
  try {
    const checkText = await callGemini(
      [{ text: `Is "${safeQuery}" a food, dish, ingredient, or cooking topic? Reply with just YES or NO.` }],
      { maxTokens: 5, temperature: 0.1, model: MODEL_CHECK }
    );
    if (!checkText.trim().toUpperCase().startsWith('YES')) {
      return res.status(400).json({
        error: 'not_food',
        message: 'Please search for a food or recipe topic — FreshPlate is your kitchen companion! 🍽️'
      });
    }
  } catch (err) {
    console.error('[search-recipes/check]', err.message);
    // If the safety check itself errors, let the query through — generation prompt is harmless
  }

  // Stage 3 — generate recipes
  // Index 0 MUST be the exact dish the user typed. Indexes 1-3 are similar but distinct dishes.
  let recipePrompt =
    `The user searched for: "${safeQuery}". Return exactly 4 recipes as a JSON array.\n\n` +
    `ARRAY INDEX 0 — THE EXACT RECIPE THE USER SEARCHED FOR:\n` +
    `  • "name" MUST be "${safeQuery}" with proper capitalisation of each word\n` +
    `  • Give the classic, definitive version of this exact dish\n` +
    `  • DO NOT rename it, reinterpret it, or replace it with a similar dish\n` +
    `  • If the user typed "Butter Chicken", index 0 name is "Butter Chicken" — not "Chicken Curry"\n` +
    `  • If the user typed "pasta", index 0 name is "Pasta" — not "Spaghetti Bolognese"\n\n` +
    `ARRAY INDEXES 1, 2, 3 — SIMILAR RECIPES (genuinely different dishes):\n` +
    `  • Share the same cuisine, main ingredient, or cooking method as "${safeQuery}"\n` +
    `  • Must each be a clearly different recipe from index 0 and from each other\n` +
    `  • Example: "${safeQuery}" = "Chicken Stir Fry" → indexes 1-3: "Beef and Broccoli", "Shrimp Fried Rice", "Pad Thai"\n\n`;

  if (Array.isArray(pantryItems) && pantryItems.length > 0) {
    const items = pantryItems.slice(0, 10).map(i => sanitize(String(i?.name ?? i), 50)).join(', ');
    recipePrompt +=
      `Where it fits naturally, use these pantry items: ${items}.\n\n`;
  }

  recipePrompt +=
    `Return ONLY a JSON array (no markdown, no extra text):\n` +
    `[{"name":"str","emoji":"str","prepTime":"20 min","difficulty":"Easy|Medium|Hard","calories":0,"protein":0.0,"carbs":0.0,"fat":0.0,"ingredients":["str"],"steps":["str"]}]\n` +
    `Include 4-7 ingredients with amounts and 4-6 clear cooking steps. Accurate macros per serving.`;

  try {
    let text = await callGemini([{ text: recipePrompt }], { maxTokens: 5000 });
    let parsed;
    try {
      parsed = JSON.parse(extractJSON(text));
    } catch (_) {
      // JSON truncated — retry once on flash with same prompt
      console.warn('[search-recipes] JSON truncated, retrying on flash');
      text   = await callGemini([{ text: recipePrompt }], { maxTokens: 5000, model: 'gemini-2.5-flash' });
      parsed = JSON.parse(extractJSON(text));
    }
    if (!Array.isArray(parsed)) throw new Error('Unexpected response format from AI.');

    // Guarantee the exact-match recipe is first — reorder if the AI drifted
    const norm = s => s.toLowerCase().replace(/[^a-z0-9\s]/g, '').trim();
    const queryNorm = norm(safeQuery);
    const exactIdx = parsed.findIndex(r => {
      const n = norm(r.name || '');
      return n === queryNorm || n.includes(queryNorm) || queryNorm.includes(n);
    });
    if (exactIdx > 0) {
      const [match] = parsed.splice(exactIdx, 1);
      parsed.unshift(match);
    }
    // Force the first recipe's name to exactly match the query (proper title case)
    if (parsed.length > 0) {
      const titleCase = safeQuery.replace(/\b\w/g, c => c.toUpperCase());
      parsed[0].name = titleCase;
    }

    setCachedRecipes(cacheKey, parsed);
    res.json(parsed);
  } catch (err) {
    console.error('[search-recipes]', err.message);
    res.status(500).json({ error: err.message });
  }
});

// ---------- Generate Single Recipe ----------

app.post('/api/ai/generate-single-recipe', ...aiLimiter, async (req, res) => {
  const { name } = req.body;
  console.log('[generate-single-recipe] name:', name);
  if (!name) return res.status(400).json({ error: 'name is required' });

  const safeName = sanitize(name, 100);
  const prompt =
    `Generate a recipe for: ${safeName}.\n` +
    'Return ONLY compact JSON (no markdown): ' +
    '{"emoji":"str","calories":0,"protein":0.0,"carbs":0.0,"fat":0.0,"ingredients":["str"],"steps":["str"]}\n' +
    'Include 5-8 ingredients with quantities and 4-6 clear cooking steps. Accurate nutrition per serving.';

  try {
    const text = await callGemini([{ text: prompt }], { maxTokens: 800 });
    res.json(JSON.parse(extractJSON(text)));
  } catch (err) {
    console.error('[generate-single-recipe]', err.message);
    res.status(500).json({ error: err.message });
  }
});

// ---------- Generate Meal Plan ----------

app.post('/api/ai/generate-meal-plan', ...aiLimiter, mealPlanDailyLimiter, async (req, res) => {
  console.log('[generate-meal-plan] request received');
  const {
    days = 7, targetCalories = 2000,
    proteinGoal = 150, carbGoal = 200, fatGoal = 65,
    pantryItems = [], budget = 0, cuisines = []
  } = req.body;

  const safeDays    = Math.min(Math.max(Number(days) || 7, 1), 7);
  const safeCals    = Math.min(Math.max(Number(targetCalories) || 2000, 800), 5000);
  const safeProtein = Number(proteinGoal) || 150;
  const safeCarbs   = Number(carbGoal)    || 200;
  const safeFat     = Number(fatGoal)     || 65;
  const safeBudget  = Number(budget)      || 0;

  let prompt = `Create a ${safeDays}-day meal plan. Daily targets: ${safeCals}kcal, ` +
               `${Math.round(safeProtein)}g protein, ${Math.round(safeCarbs)}g carbs, ${Math.round(safeFat)}g fat.`;

  if (Array.isArray(cuisines) && cuisines.length > 0) {
    prompt += ` Cuisines to rotate: ${cuisines.slice(0, 5).map(c => sanitize(String(c), 30)).join(', ')}.`;
  }
  if (Array.isArray(pantryItems) && pantryItems.length > 0) {
    const hasExpiry = pantryItems.some(i => i?.expiresInDays !== undefined && i.expiresInDays >= 0);
    const items = pantryItems.map(i => {
      const name = sanitize(String(i?.name ?? i), 50);
      return (hasExpiry && i?.expiresInDays !== undefined && i.expiresInDays >= 0)
        ? `${name} (expires in ${i.expiresInDays}d)` : name;
    }).join(', ');
    if (hasExpiry) {
      prompt += ` The user already has these pantry items — USE ALL OF THEM in recipes, prioritising those expiring soonest: ${items}. Build every meal around what is already available. Any additional ingredients NOT in this list must be minimal and kept within the grocery budget.`;
    } else {
      prompt += ` The user already has these pantry items — USE ALL OF THEM across the meal plan: ${items}. Build every meal primarily around these ingredients. Any extra ingredients not in this list are what the user will need to buy.`;
    }
  }
  if (safeBudget > 0) {
    const pantryCount = Array.isArray(pantryItems) ? pantryItems.length : 0;
    if (pantryCount > 0) {
      prompt += ` Weekly grocery budget for additional ingredients (not already in pantry): ~$${Math.round(safeBudget)}. Keep the total cost of non-pantry ingredients within this budget.`;
    } else {
      prompt += ` Weekly grocery budget: ~$${Math.round(safeBudget)}.`;
    }
  }

  prompt +=
    '\nReturn ONLY compact JSON:\n' +
    '{"name":"str","description":"str","emoji":"🍽️","color":"AccentGreen","days":[{"dayNumber":1,"meals":[' +
    '{"mealType":"Breakfast","name":"str","calories":0,"protein":0.0,"carbs":0.0,"fat":0.0,"fiber":0.0,"servingSize":"str","emoji":"str"}]}]}\n' +
    'mealType: Breakfast, Lunch, Dinner, or Snack. Include all 4 per day. Realistic macros. Vary across days.';

  try {
    const planTokens = Math.min(Math.max(Math.ceil(safeDays * 700), 4000), 6000);

    let text = await callGemini([{ text: prompt }], { maxTokens: planTokens, temperature: 0.6 });
    let parsed;
    try {
      parsed = JSON.parse(extractJSON(text));
    } catch (_) {
      // JSON was truncated — retry once with a shorter, no-fiber prompt on flash
      console.warn('[generate-meal-plan] JSON truncated, retrying with compact prompt');
      const compactPrompt = prompt.replace(
        '{"mealType":"Breakfast","name":"str","calories":0,"protein":0.0,"carbs":0.0,"fat":0.0,"fiber":0.0,"servingSize":"str","emoji":"str"}',
        '{"mealType":"Breakfast","name":"str","calories":0,"protein":0.0,"carbs":0.0,"fat":0.0,"servingSize":"str","emoji":"str"}'
      );
      text   = await callGemini([{ text: compactPrompt }], { maxTokens: planTokens, temperature: 0.6 });
      parsed = JSON.parse(extractJSON(text));
    }
    res.json(parsed);
  } catch (err) {
    console.error('[generate-meal-plan]', err.message);
    const friendly = (err.message.includes('503') || err.message.includes('429'))
      ? 'The AI is currently busy with multiple requests. Please try again in 30 seconds.'
      : err.message;
    res.status(500).json({ error: friendly });
  }
});

// ---------- Personalized Notifications ----------

app.post('/api/ai/personalize-notifications', ...aiLimiter, notifDailyLimiter, async (req, res) => {
  console.log('[personalize-notifications] request received');
  const {
    name           = 'there',
    dayOfWeek      = 'today',
    streak         = 0,
    todayMealsLogged = 0,
    expiringItems  = [],
    groceryCount   = 0,
    groceryPreview = '',
    upcomingReminders = [],
    friendsCount   = 0
  } = req.body;

  const safeName = sanitize(String(name), 40);

  // Build the expiry summary
  const expirySummary = expiringItems.slice(0, 5)
    .map(i => `${sanitize(String(i.name), 40)} (${i.daysUntilExpiry} day${i.daysUntilExpiry !== 1 ? 's' : ''})`)
    .join(', ') || 'none';

  const reminderSummary = upcomingReminders.slice(0, 3)
    .map(r => sanitize(String(r), 60)).join(', ') || 'none';

  const prompt =
    `You are the notification engine for FreshPlate, a personal meal-planning and food-tracking app.\n` +
    `Generate exactly 3-4 personalized push notifications for this user. Be warm, specific, and actionable.\n\n` +
    `USER DATA:\n` +
    `- Name: ${safeName}\n` +
    `- Day: ${sanitize(dayOfWeek, 20)}\n` +
    `- Streak: ${Math.max(0, Number(streak) || 0)} day(s)\n` +
    `- Meals logged today: ${Number(todayMealsLogged) || 0} of 4\n` +
    `- Expiring pantry items: ${expirySummary}\n` +
    `- Grocery list: ${Number(groceryCount) || 0} item(s) ${sanitize(groceryPreview, 80)}\n` +
    `- Today's scheduled reminders: ${reminderSummary}\n` +
    `- Friends on app: ${Number(friendsCount) || 0}\n\n` +
    `Return ONLY compact JSON:\n` +
    `{"notifications":[{"id":"str","title":"str","body":"str","hour":9,"minute":0,"category":"expiry|streak|grocery|meal|friend"}]}\n\n` +
    `RULES (follow strictly):\n` +
    `- Generate exactly 3-4 notifications, never more\n` +
    `- Spread times: morning 8-10h, midday 12-13h, evening 17-19h, night 20-21h\n` +
    `- ONLY include a category if there is actual data for it (skip expiry if none, skip grocery if count=0, skip friend if 0)\n` +
    `- expiry: name the specific item and suggest a recipe or use for it\n` +
    `- streak: if streak > 0 motivate continuation; if streak = 0 encourage starting; include day count\n` +
    `- grocery: mention 1-2 specific item names from the preview\n` +
    `- meal: only if todayMealsLogged < 2, encourage logging to protect the streak\n` +
    `- friend: only if friendsCount > 0, mention comparing progress\n` +
    `- Use the user's first name naturally in 1 notification (not all)\n` +
    `- Title max 50 chars, body max 110 chars\n` +
    `- Friendly and encouraging tone, never guilt-tripping`;

  try {
    const text   = await callGemini([{ text: prompt }], { maxTokens: 1500, temperature: 0.8 });
    const parsed = JSON.parse(extractJSON(text));

    // Validate shape
    const notifs = (parsed.notifications ?? []).slice(0, 5);
    if (!Array.isArray(notifs)) throw new Error('Invalid notifications array');

    res.json({ notifications: notifs });
  } catch (err) {
    console.error('[personalize-notifications]', err.message);
    res.status(500).json({ error: err.message });
  }
});

// ---------- Scan Pantry Item (food photo → name + expiry estimate) ----------

app.post('/api/ai/scan-pantry-item', ...aiLimiter, async (req, res) => {
  const { image, mimeType = 'image/jpeg' } = req.body;
  console.log('[scan-pantry-item] request received');
  if (!image) return res.status(400).json({ error: 'image is required' });

  try {
    const text = await callGemini(
      [
        { inlineData: { mimeType, data: image } },
        {
          text: 'Identify the food item in this image. Estimate how many days it stays fresh (expiryDays). ' +
                'Examples: fresh chicken=3, berries=3, milk=7, bread=5, apple=14, hard cheese=30, canned goods=365. ' +
                'Return ONLY compact JSON with no extra text or markdown: ' +
                '{"name":"str","emoji":"str","quantity":1,"unit":"pcs","category":"produce|proteins|dairy|grains|condiments|snacks|beverages|frozen|other","expiryDays":7}'
        }
      ],
      { maxTokens: 200, temperature: 0.2 }
    );
    res.json(JSON.parse(extractJSON(text)));
  } catch (err) {
    console.error('[scan-pantry-item]', err.message);
    res.status(500).json({ error: err.message });
  }
});

// ---------- Scan Receipt ----------

app.post('/api/ai/scan-receipt', ...aiLimiter, async (req, res) => {
  const { image, mimeType = 'image/jpeg' } = req.body;
  console.log('[scan-receipt] request received');
  if (!image) return res.status(400).json({ error: 'image is required' });

  try {
    const text = await callGemini(
      [
        { inlineData: { mimeType, data: image } },
        {
          text: 'This is a grocery receipt. Extract every food and grocery item purchased. ' +
                'For each item return name, emoji, quantity, unit, and category. ' +
                'Return ONLY a JSON array with no extra text or markdown: ' +
                '[{"name":"str","emoji":"str","quantity":1,"unit":"pcs","category":"produce|proteins|dairy|grains|condiments|snacks|beverages|frozen|other"}] ' +
                'Skip store names, taxes, totals, dates, and non-food items. If no items found return [].'
        }
      ],
      { maxTokens: 1000, temperature: 0.2 }
    );
    const parsed = JSON.parse(extractJSON(text));
    res.json({ items: Array.isArray(parsed) ? parsed : [] });
  } catch (err) {
    console.error('[scan-receipt]', err.message);
    res.status(500).json({ error: err.message });
  }
});

// ---------- Scan Groceries / Fridge (photo of multiple items) ----------

app.post('/api/ai/scan-groceries', ...aiLimiter, async (req, res) => {
  const { image, mimeType = 'image/jpeg', mode = 'groceries' } = req.body;
  console.log('[scan-groceries] request received, mode:', mode);
  if (!image) return res.status(400).json({ error: 'image is required' });

  const isFridge = mode === 'fridge';
  const promptText = isFridge
    ? 'Look at everything visible inside this refrigerator. Identify every distinct food item stored on shelves, in drawers, and in the door. ' +
      'For each item estimate a realistic quantity based on what you can see. ' +
      'Return ONLY a JSON array with no extra text or markdown: ' +
      '[{"name":"str","emoji":"str","quantity":1.0,"unit":"pcs","category":"produce|proteins|dairy|grains|condiments|snacks|beverages|frozen|other"}] ' +
      'Use sensible units: pcs for countable items, g/kg for loose items, L/mL for liquids. ' +
      'If no food items are visible return [].'
    : 'Look at this photo of groceries. Identify every distinct food item visible. ' +
      'For each item estimate the quantity based on what you can see (count individual pieces, estimate weight/volume for bulk). ' +
      'Return ONLY a JSON array with no extra text or markdown: ' +
      '[{"name":"str","emoji":"str","quantity":1.0,"unit":"pcs","category":"produce|proteins|dairy|grains|condiments|snacks|beverages|frozen|other"}] ' +
      'Use sensible units: pcs for countable items, g/kg for loose items, L/mL for liquids, bags/boxes for packaged items. ' +
      'If no food items are visible return [].';

  try {
    const text = await callGemini(
      [{ inlineData: { mimeType, data: image } }, { text: promptText }],
      { maxTokens: 800, temperature: 0.2 }
    );
    const parsed = JSON.parse(extractJSON(text));
    res.json({ items: Array.isArray(parsed) ? parsed : [] });
  } catch (err) {
    console.error('[scan-groceries]', err.message);
    res.status(500).json({ error: err.message });
  }
});

// ---------- Weekly Nutrition Report ----------

// ---------- Cook Tonight (suggest meals from pantry) ----------

app.post('/api/ai/suggest-from-pantry', ...aiLimiter, async (req, res) => {
  const { pantryItems = [], restrictions = [], count = 2 } = req.body;

  const safeCount = Math.min(Math.max(Number(count) || 2, 1), 3);
  const items = (Array.isArray(pantryItems) ? pantryItems : []).slice(0, 20)
    .map(i => {
      const name = sanitize(String(i?.name ?? i), 40);
      return (i?.expiresInDays !== undefined && i.expiresInDays >= 0)
        ? `${name} (expires in ${i.expiresInDays}d)`
        : name;
    })
    .join(', ');

  if (!items) return res.status(400).json({ error: 'pantryItems is required' });

  const restrictionStr = Array.isArray(restrictions) && restrictions.length > 0
    ? ` Dietary restrictions: ${restrictions.slice(0, 5).map(r => sanitize(String(r), 30)).join(', ')}.`
    : '';

  const prompt =
    `I have these pantry items: ${items}.${restrictionStr}\n` +
    `Suggest ${safeCount} complete meals I can make tonight using primarily these ingredients. ` +
    'Prioritize items expiring soonest. Each meal should be practical and quick.\n' +
    'Return ONLY compact JSON:\n' +
    '{"suggestions":[{"name":"str","emoji":"str","description":"str","keyIngredients":["str"],"cookTime":"str"}]}\n' +
    'description: 1 engaging sentence. keyIngredients: 3-4 main pantry items used. cookTime: e.g. "20 min".';

  try {
    const text = await callGemini([{ text: prompt }], { maxTokens: 600, temperature: 0.7 });
    res.json(JSON.parse(extractJSON(text)));
  } catch (err) {
    console.error('[suggest-from-pantry]', err.message);
    res.status(500).json({ error: err.message });
  }
});

// ---------- Weekly Nutrition Report ----------

const weeklyReportLimiter = rateLimit({
  windowMs: 24 * 60 * 60 * 1000,
  max: 3,
  standardHeaders: true,
  legacyHeaders: false,
  validate: { xForwardedForHeader: false },
  message: { error: 'Weekly report limit reached for today.' }
});

app.post('/api/ai/weekly-nutrition-report', ...aiLimiter, weeklyReportLimiter, async (req, res) => {
  console.log('[weekly-nutrition-report] request received');
  const { name = 'there', weeklyData = [], goals = {}, pantry = {}, daysNotLogged = 0 } = req.body;

  const safeName          = sanitize(String(name), 40);
  const calGoal           = Number(goals.calories)               || 2000;
  const proGoal           = Number(goals.protein)                || 150;
  const expiredThisWeek   = Math.max(0, Number(pantry.expiredThisWeek)       || 0);
  const expiringNextThree = Math.max(0, Number(pantry.expiringNextThreeDays) || 0);
  const missedDays        = Math.max(0, Math.min(7, Number(daysNotLogged)    || 0));

  // Build a readable summary of the week
  const daySummaries = weeklyData.slice(0, 7).map(d => {
    const cal = Number(d.calories) || 0;
    const pro = Number(d.protein)  || 0;
    const day = sanitize(String(d.dayName || ''), 15);
    return cal === 0
      ? `${day}: no food logged`
      : `${day}: ${cal} kcal, ${Math.round(pro)}g protein`;
  }).join('\n');

  const daysHitCalories = weeklyData.filter(d => {
    const cal = Number(d.calories) || 0;
    return cal >= calGoal * 0.85 && cal <= calGoal * 1.15;
  }).length;

  const daysHitProtein = weeklyData.filter(d => Number(d.protein) >= proGoal * 0.9).length;

  const pantryLine = expiredThisWeek > 0
    ? `Pantry: ${expiredThisWeek} item${expiredThisWeek > 1 ? 's' : ''} expired this week (food wasted)`
    : `Pantry: no items expired this week (great job using food up!)`;
  const expiryWarning = expiringNextThree > 0
    ? `${expiringNextThree} item${expiringNextThree > 1 ? 's' : ''} expiring in the next 3 days`
    : `No items expiring in the next 3 days`;

  const prompt =
    `You are the nutrition coach inside FreshPlate, a meal-tracking and pantry app.\n` +
    `Write a personalized weekly summary for ${safeName} covering nutrition AND food waste.\n\n` +
    `WEEKLY NUTRITION (last 7 days):\n${daySummaries}\n\n` +
    `GOALS: ${calGoal} kcal/day, ${proGoal}g protein/day\n` +
    `Hit calorie target: ${daysHitCalories}/7 days\n` +
    `Hit protein target: ${daysHitProtein}/7 days\n` +
    `Days with no food logged: ${missedDays}/7\n\n` +
    `PANTRY & FOOD WASTE:\n` +
    `${pantryLine}\n` +
    `${expiryWarning}\n\n` +
    `Return ONLY compact JSON (no markdown):\n` +
    `{"headline":"str","summary":"str","highlights":["str","str","str"],"tip":"str"}\n\n` +
    `RULES:\n` +
    `- headline: short punchy title, max 8 words\n` +
    `- summary: 2 sentences — one on nutrition, one on pantry/food waste; warm and specific with actual numbers\n` +
    `- highlights: exactly 3 insights covering a mix of: nutrition wins/misses, logging consistency, food waste or savings\n` +
    `- tip: one concrete actionable tip for next week (could be nutrition OR reducing food waste)\n` +
    `- If items expired, gently acknowledge it and suggest a fix; if none expired, celebrate it\n` +
    `- If logging was missed, note it encouragingly without guilt\n` +
    `- Never mention "AI", "Gemini", or "generated" — write as a real coach\n` +
    `- Friendly, encouraging tone; never guilt-tripping`;

  try {
    const text   = await callGemini([{ text: prompt }], { maxTokens: 600, temperature: 0.7 });
    const parsed = JSON.parse(extractJSON(text));
    res.json({
      headline:   sanitize(String(parsed.headline  || ''), 120),
      summary:    sanitize(String(parsed.summary   || ''), 400),
      highlights: (Array.isArray(parsed.highlights) ? parsed.highlights : []).slice(0, 3).map(h => sanitize(String(h), 200)),
      tip:        sanitize(String(parsed.tip        || ''), 200)
    });
  } catch (err) {
    console.error('[weekly-nutrition-report]', err.message);
    res.status(500).json({ error: err.message });
  }
});

// ---------- Food Image (Pexels) ----------

// Persistent URL cache — survives server restarts, URLs are tiny
const IMAGE_CACHE_FILE = '/tmp/food_image_cache.json';
let foodImageCache = new Map();
try {
  const saved = JSON.parse(readFileSync(IMAGE_CACHE_FILE, 'utf8'));
  foodImageCache = new Map(Object.entries(saved));
  console.log(`[image-cache] loaded ${foodImageCache.size} cached images`);
} catch { /* first run */ }

function persistImageCache() {
  try {
    writeFileSync(IMAGE_CACHE_FILE, JSON.stringify(Object.fromEntries(foodImageCache)));
  } catch (e) { console.error('[image-cache] persist error:', e.message); }
}

async function fetchPexelsImageUrl(name) {
  const apiKey = process.env.PEXELS_API_KEY;
  if (!apiKey) throw new Error('PEXELS_API_KEY not set');

  const cacheKey = name.toLowerCase().trim();
  if (foodImageCache.has(cacheKey)) return foodImageCache.get(cacheKey);

  const query = encodeURIComponent(`${sanitize(name, 80)} food`);
  const res = await fetch(
    `https://api.pexels.com/v1/search?query=${query}&per_page=1&orientation=square`,
    { headers: { Authorization: apiKey } }
  );
  if (!res.ok) throw new Error(`Pexels API ${res.status}`);

  const data = await res.json();
  const url = data.photos?.[0]?.src?.medium;
  if (!url) throw new Error('No photo found on Pexels');

  foodImageCache.set(cacheKey, url);
  persistImageCache();
  return url;
}

app.post('/api/images/food', async (req, res) => {
  const { name } = req.body ?? {};
  if (!name || typeof name !== 'string') return res.status(400).json({ error: 'name is required' });

  console.log('[images/food] request:', name);
  try {
    const imageUrl = await fetchPexelsImageUrl(name);
    res.json({ imageUrl });
  } catch (err) {
    console.error('[images/food]', err.message);
    res.status(500).json({ error: 'Could not fetch food image.' });
  }
});

// ---------- Error handler ----------

app.use((err, _req, res, _next) => {
  console.error('[unhandled]', err);
  res.status(500).json({ error: 'Internal server error' });
});

app.listen(PORT, () => {
  console.log(`FreshPlate backend running on port ${PORT}`);
  if (GEMINI_KEYS.length === 0) {
    console.warn('WARNING: No GEMINI_API_KEY(S) configured — AI endpoints will fail.');
  } else {
    const concLimit = Number(process.env.GEMINI_MAX_CONCURRENT) || 20;
    console.log(`Gemini: ${GEMINI_KEYS.length} key(s) loaded, concurrency limit ${concLimit}`);
  }
});
