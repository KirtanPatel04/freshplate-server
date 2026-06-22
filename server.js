import 'dotenv/config';
import express from 'express';
import rateLimit from 'express-rate-limit';
import { createClient } from '@supabase/supabase-js';
import { readFileSync, writeFileSync } from 'fs';

const app  = express();
const PORT = process.env.PORT || 3000;

// ---------- Middleware ----------

app.set('trust proxy', 1);
app.use(express.json({ limit: '25mb' }));

// ── Rate limiters ──────────────────────────────────────────────────────────────
// Tier 1 — per-minute burst guard (all AI endpoints)
const perMinuteLimiter = rateLimit({
  windowMs: 60_000,
  max: 30,
  standardHeaders: true,
  legacyHeaders: false,
  validate: { xForwardedForHeader: false },
  message: { error: 'Too many requests. Please slow down.' }
});

// Tier 2 — per-day overall AI budget (200 calls/IP/day)
const dailyLimiter = rateLimit({
  windowMs: 24 * 60 * 60 * 1000,
  max: 200,
  standardHeaders: true,
  legacyHeaders: false,
  validate: { xForwardedForHeader: false },
  message: { error: 'Daily limit reached. Come back tomorrow!' }
});

// Tier 3 — strict daily limits on expensive endpoints
const mealPlanDailyLimiter = rateLimit({
  windowMs: 24 * 60 * 60 * 1000,
  max: 5,
  standardHeaders: true,
  legacyHeaders: false,
  validate: { xForwardedForHeader: false },
  message: { error: 'Meal plan limit: 5 per day. Come back tomorrow!' }
});

const recipeDailyLimiter = rateLimit({
  windowMs: 24 * 60 * 60 * 1000,
  max: 30,
  standardHeaders: true,
  legacyHeaders: false,
  validate: { xForwardedForHeader: false },
  message: { error: 'Recipe search limit reached for today.' }
});

const notifDailyLimiter = rateLimit({
  windowMs: 24 * 60 * 60 * 1000,
  max: 10,
  standardHeaders: true,
  legacyHeaders: false,
  validate: { xForwardedForHeader: false },
  message: { error: 'Notification generation limit reached for today.' }
});


// Stack the two tiers: every AI route uses perMinute + daily
const aiLimiter = [perMinuteLimiter, dailyLimiter];

// ---------- Gemini helpers ----------

const GEMINI_BASE = 'https://generativelanguage.googleapis.com/v1beta/models';
const MODEL_MAIN  = 'gemini-2.5-flash-lite';   // fast + cheap — retry handles 503 spikes
const MODEL_CHECK = 'gemini-2.5-flash-lite';  // cheapest — yes/no safety check only

async function callGemini(parts, { maxTokens = 8000, temperature = 0.7, model = MODEL_MAIN } = {}) {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) throw new Error('GEMINI_API_KEY is not set on the server.');

  const body = JSON.stringify({
    contents: [{ role: 'user', parts }],
    generationConfig: { maxOutputTokens: maxTokens, temperature }
  });

  for (let attempt = 1; attempt <= 3; attempt++) {
    const res = await fetch(`${GEMINI_BASE}/${model}:generateContent?key=${apiKey}`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body
    });

    if ((res.status === 503 || res.status === 429) && attempt < 3) {
      await new Promise(r => setTimeout(r, attempt * 1500));
      continue;
    }

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Gemini API ${res.status}: ${text}`);
    }

    const data = await res.json();
    return data.candidates?.[0]?.content?.parts?.[0]?.text ?? '';
  }
}

// Extract first JSON object or array from text (tolerates model preamble)
function extractJSON(text) {
  const arrStart = text.indexOf('[');
  const objStart = text.indexOf('{');

  if (arrStart !== -1 && (objStart === -1 || arrStart < objStart)) {
    const end = text.lastIndexOf(']');
    if (end !== -1) return text.slice(arrStart, end + 1);
  }
  if (objStart !== -1) {
    const end = text.lastIndexOf('}');
    if (end !== -1) return text.slice(objStart, end + 1);
  }
  return text;
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

// POST /auth/forgot-password  — triggers a Supabase password reset email
app.post('/auth/forgot-password', requireSupabase, async (req, res) => {
  try {
    const { email } = req.body ?? {};
    if (!email) return res.status(400).json({ error: 'Email required.' });

    // Always return success to prevent email enumeration attacks
    await supabase.auth.resetPasswordForEmail(email.toLowerCase().trim(), {
      redirectTo: 'freshplate://reset-password'
    });
    res.json({ message: 'If an account exists for this email, a reset link has been sent.' });
  } catch (err) {
    console.error('[auth/forgot-password]', err);
    res.status(500).json({ error: 'Server error. Please try again.' });
  }
});

// ---------- Health ----------

app.get('/health', (_req, res) => {
  console.log('[health] ping');
  res.json({ ok: true });
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
    const text   = await callGemini([{ text: recipePrompt }], { maxTokens: 16000 });
    const parsed = JSON.parse(extractJSON(text));
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

  const safeDays    = Math.min(Math.max(Number(days) || 7, 1), 14);
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
    const text = await callGemini([{ text: prompt }], { maxTokens: 16000, temperature: 0.6 });
    res.json(JSON.parse(extractJSON(text)));
  } catch (err) {
    console.error('[generate-meal-plan]', err.message);
    res.status(500).json({ error: err.message });
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
  const { name = 'there', weeklyData = [], goals = {} } = req.body;

  const safeName = sanitize(String(name), 40);
  const calGoal  = Number(goals.calories) || 2000;
  const proGoal  = Number(goals.protein)  || 150;

  // Build a readable summary of the week
  const daySummaries = weeklyData.slice(0, 7).map(d => {
    const cal = Number(d.calories) || 0;
    const pro = Number(d.protein)  || 0;
    const day = sanitize(String(d.dayName || ''), 15);
    return `${day}: ${cal} kcal, ${Math.round(pro)}g protein`;
  }).join('\n');

  const daysHitCalories = weeklyData.filter(d => {
    const cal = Number(d.calories) || 0;
    return cal >= calGoal * 0.85 && cal <= calGoal * 1.15;
  }).length;

  const daysHitProtein = weeklyData.filter(d => Number(d.protein) >= proGoal * 0.9).length;

  const prompt =
    `You are the nutrition coach inside FreshPlate, a meal-tracking app.\n` +
    `Write a personalized weekly nutrition summary for ${safeName}.\n\n` +
    `WEEKLY DATA (last 7 days):\n${daySummaries}\n\n` +
    `GOALS: ${calGoal} kcal/day, ${proGoal}g protein/day\n` +
    `Hit calorie target: ${daysHitCalories}/7 days\n` +
    `Hit protein target: ${daysHitProtein}/7 days\n\n` +
    `Return ONLY compact JSON (no markdown):\n` +
    `{"headline":"str","summary":"str","highlights":["str","str","str"],"tip":"str"}\n\n` +
    `RULES:\n` +
    `- headline: short punchy title, max 8 words, e.g. "Strong week — protein was your superpower"\n` +
    `- summary: 2 sentences, warm and specific, mention actual numbers\n` +
    `- highlights: exactly 3 bullet-point insights (what went well, what to watch, a pattern noticed)\n` +
    `- tip: one concrete actionable tip for next week\n` +
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
  if (!process.env.GEMINI_API_KEY) {
    console.warn('WARNING: GEMINI_API_KEY is not set — AI endpoints will fail.');
  }
});
