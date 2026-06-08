import 'dotenv/config';
import express from 'express';
import rateLimit from 'express-rate-limit';

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

const foodImageDailyLimiter = rateLimit({
  windowMs: 24 * 60 * 60 * 1000,
  max: 40,
  standardHeaders: true,
  legacyHeaders: false,
  validate: { xForwardedForHeader: false },
  message: { error: 'Food image generation limit reached for today.' }
});

// Stack the two tiers: every AI route uses perMinute + daily
const aiLimiter = [perMinuteLimiter, dailyLimiter];

// ---------- Gemini helper ----------

const GEMINI_URL =
  'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent';

async function callGemini(parts, { maxTokens = 8000, temperature = 0.7 } = {}) {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) throw new Error('GEMINI_API_KEY is not set on the server.');

  const res = await fetch(`${GEMINI_URL}?key=${apiKey}`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      contents: [{ role: 'user', parts }],
      generationConfig: { maxOutputTokens: maxTokens, temperature }
    })
  });

  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Gemini API ${res.status}: ${body}`);
  }

  const data = await res.json();
  return data.candidates?.[0]?.content?.parts?.[0]?.text ?? '';
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
  // Kept separate so the recipe generation prompt is never confused by branching logic
  try {
    const checkText = await callGemini(
      [{ text: `Is "${safeQuery}" a food, dish, ingredient, or cooking topic? Reply with just YES or NO.` }],
      { maxTokens: 5, temperature: 0.1 }
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
  // The query is the hard constraint — pantry items are a soft hint only.
  // Never substitute a different dish just because pantry items are present.
  let recipePrompt =
    `Generate exactly 4 recipe variations specifically for: "${safeQuery}".\n` +
    `CRITICAL RULE: Every single recipe MUST be a version of "${safeQuery}". ` +
    `Do NOT replace it with a different dish — not even if pantry items suggest it.\n`;

  if (Array.isArray(pantryItems) && pantryItems.length > 0) {
    const items = pantryItems.slice(0, 10).map(i => sanitize(String(i?.name ?? i), 50)).join(', ');
    recipePrompt +=
      `If any of these pantry items fit naturally into a "${safeQuery}" recipe, include them — ` +
      `but only if they belong in that dish: ${items}.\n`;
  }

  recipePrompt +=
    `Return ONLY a JSON array with exactly 4 recipes (no markdown, no extra text):\n` +
    `[{"name":"str","emoji":"str","prepTime":"20 min","difficulty":"Easy|Medium|Hard","calories":0,"protein":0.0,"carbs":0.0,"fat":0.0,"ingredients":["str"],"steps":["str"]}]\n` +
    `Include 4-7 ingredients with amounts and 4-6 clear cooking steps. Accurate macros per serving.`;

  try {
    const text   = await callGemini([{ text: recipePrompt }], { maxTokens: 3000 });
    const parsed = JSON.parse(extractJSON(text));
    if (!Array.isArray(parsed)) throw new Error('Unexpected response format from AI.');
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
    const items = pantryItems.slice(0, 12).map(i => sanitize(String(i?.name ?? i), 50)).join(', ');
    prompt += ` Prioritise pantry items: ${items}.`;
  }
  if (safeBudget > 0) prompt += ` Budget: ~$${Math.round(safeBudget)}.`;

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

// ---------- Generate Food Image (Imagen 3 Fast) ----------

app.post('/api/ai/food-image', perMinuteLimiter, foodImageDailyLimiter, async (req, res) => {
  const { name } = req.body;
  console.log('[food-image] Generating image for:', name);
  if (!name) return res.status(400).json({ error: 'name is required' });

  const safeName = sanitize(name, 100);
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) return res.status(500).json({ error: 'GEMINI_API_KEY not set' });

  try {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/imagen-3.0-fast-generate-001:predict?key=${apiKey}`;
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        instances: [{
          prompt: `Professional food photography of ${safeName}. Appetizing, restaurant-quality plating, natural lighting, close-up shot on a clean white plate or wooden surface. No text, no watermarks.`
        }],
        parameters: { sampleCount: 1 }
      })
    });

    const data = await response.json();
    if (!response.ok) throw new Error(data.error?.message || `Imagen API ${response.status}`);

    const imageBase64 = data.predictions?.[0]?.bytesBase64Encoded;
    if (!imageBase64) throw new Error('No image data returned from Imagen');

    res.json({ imageData: imageBase64, mimeType: data.predictions[0].mimeType || 'image/png' });
  } catch (err) {
    console.error('[food-image]', err.message);
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
