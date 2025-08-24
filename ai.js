
/*
  ai.js — Real in‑browser AI using Transformers.js (no API keys, free)
  - Text: zero‑shot classification (Xenova/distilbert-base-uncased-mnli)
  - Image: image classification (Xenova/vit-base-patch16-224)
  - Works on static hosting (e.g., GitHub Pages). First call downloads models (cached).
*/

;(() => {
  // Exported namespace
  const EcoAI = {}
  window.EcoAI = EcoAI

  // Lazy-loaded pipelines
  let _pipelinesPromise = null

  async function ensurePipelines () {
    if (_pipelinesPromise) return _pipelinesPromise
    _pipelinesPromise = (async () => {
      // Load Transformers.js from CDN if not already present
      if (!window.transformers) {
        await new Promise((resolve, reject) => {
          const s = document.createElement('script')
          s.src = 'https://cdn.jsdelivr.net/npm/@xenova/transformers/dist/transformers.min.js'
          s.async = true
          s.onload = resolve
          s.onerror = () => reject(new Error('Failed to load Transformers.js'))
          document.head.appendChild(s)
        })
      }

      const { pipeline, env } = window.transformers
      // Use browser cache only; don't require local models
      env.allowLocalModels = false
      // Reduce memory where possible
      env.backends.onnx.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/@xenova/transformers/dist/'
      // Pipelines
      const textClassifier = await pipeline('zero-shot-classification', 'Xenova/distilbert-base-uncased-mnli')
      const imageClassifier = await pipeline('image-classification', 'Xenova/vit-base-patch16-224')
      return { pipeline, textClassifier, imageClassifier }
    })()
    return _pipelinesPromise
  }

  // Utilities
  function clamp (n, a, b) { return Math.max(a, Math.min(b, n)) }

  function keywords (s) {
    return (s || '')
      .toLowerCase()
      .replace(/[^a-z0-9 ]+/g, ' ')
      .split(/\s+/)
      .filter(Boolean)
  }

  const EDIBLE_HINTS = new Set([
    'food','fruit','vegetable','snack','drink','beverage','meat','chicken','fish','tuna','bread','rice','pasta','milk','dairy','cheese','yogurt','chocolate','candy','juice','water','coffee','tea','cookie','biscuit','banana','apple','orange','tomato','potato'
  ])

  const NON_EDIBLE_HINTS = new Set([
    'detergent','bleach','cleaner','soap','shampoo','conditioner','lotion','cream','cosmetic','makeup','electronics','phone','laptop','battery','remote','toy','clothes','tshirt','shirt','jacket','shoe','paint','glue','pill','medicine','medication','drug','supplement'
  ])

  const TEXT_LABELS = [
    'food product','drink','fruit','vegetable','snack','dairy','meat',
    'cosmetic','cleaning product','electronic device','toy','clothing','medication','plastic packaging','battery'
  ]

  const EDIBLE_LABELS = new Set(['food product','drink','fruit','vegetable','snack','dairy','meat'])

  const NUTRITION_TABLE = {
    'fruit': { kcal: 60, protein: 0.5, fat: 0.2, carbs: 15 },
    'vegetable': { kcal: 35, protein: 2, fat: 0.2, carbs: 7 },
    'snack': { kcal: 520, protein: 7, fat: 30, carbs: 55 },
    'drink': { kcal: 45, protein: 0, fat: 0, carbs: 11 },
    'dairy': { kcal: 120, protein: 6, fat: 6, carbs: 8 },
    'meat': { kcal: 250, protein: 26, fat: 15, carbs: 0 },
    'food product': { kcal: 180, protein: 5, fat: 6, carbs: 28 }
  }

  function guessNutrition (label) {
    const key = (label || '').toLowerCase()
    const base = NUTRITION_TABLE[key] || NUTRITION_TABLE['food product']
    return {
      per100g: true,
      kcal: base.kcal,
      protein_g: base.protein,
      fat_g: base.fat,
      carbs_g: base.carbs
    }
  }

  function ecoAssessmentFromLabel (label) {
    label = (label || '').toLowerCase()
    if (['drink'].includes(label)) {
      return {
        packaging: 'Likely plastic/aluminum bottle or can',
        footprint: 'Medium (beverage processing + packaging)',
        recycle: 'Prefer aluminum can or reusable bottle'
      }
    } else if (['fruit','vegetable'].includes(label)) {
      return {
        packaging: 'Minimal; avoid plastic wrap when possible',
        footprint: 'Low to medium (transport dependent)',
        recycle: 'Compost organic waste'
      }
    } else if (['snack','dairy','food product','meat'].includes(label)) {
      return {
        packaging: 'Often plastic/laminate; hard to recycle',
        footprint: 'Medium to high (processing, cold chain for dairy/meat)',
        recycle: 'Check local rules; reduce single-use packaging'
      }
    } else if (['cosmetic','cleaning product','electronic device','battery'].includes(label)) {
      return {
        packaging: 'Plastic or mixed materials',
        footprint: 'Medium',
        recycle: 'Take to proper e-waste/hazard drop-off if applicable'
      }
    }
    return {
      packaging: 'Unknown',
      footprint: 'Unknown',
      recycle: 'Check local guidance'
    }
  }

  function alternativesFromLabel (label) {
    label = (label || '').toLowerCase()
    if (label === 'drink') return ['Use a reusable bottle', 'Choose aluminum cans over plastic when possible']
    if (label === 'snack') return ['Buy in bulk to reduce packaging', 'Choose snacks in paper or compostable wrapping']
    if (label === 'meat') return ['Try one meatless day per week', 'Choose poultry or fish over red meat for lower footprint']
    if (label === 'cosmetic') return ['Refill stations if available', 'Choose products with minimal packaging']
    if (label === 'cleaning product') return ['Concentrates or refills', 'Use vinegar/baking soda for some tasks']
    if (label === 'electronic device') return ['Repair before replacing', 'Buy refurbished where possible']
    if (label === 'battery') return ['Use rechargeable batteries', 'Recycle at e-waste centers']
    if (['fruit','vegetable','dairy','food product'].includes(label)) return ['Prefer local/seasonal options', 'Choose minimal packaging']
    return ['Consider products with eco-labels', 'Reduce single-use packaging']
  }

  function tipsFromContext (kw) {
    const kset = new Set(kw)
    if (kset.has('bottle')) return ['Carry a reusable bottle', 'Avoid single-use plastics']
    if (kset.has('battery')) return ['Collect used batteries for proper recycling']
    if (kset.has('tuna') || kset.has('fish')) return ['Look for MSC-certified seafood']
    if (kset.has('beef') || kset.has('meat')) return ['Moderate red meat intake to lower emissions']
    return ['Check your local recycling guide for specific rules']
  }

  function edibleDecision ({ textTop, imageTop, textKw }) {
    // Image/top label hint
    let label = (imageTop && imageTop.label) ? imageTop.label.toLowerCase() : null
    // Heuristics from image label
    const edibleLikelyFromImage = label && /banana|apple|orange|fruit|vegetable|food|drink|beverage|bread|pizza|burger|sandwich|milk|coffee|tea|chocolate|snack/gi.test(label)
    // Heuristics from text
    const hasEdibleKw = textKw.some(k => EDIBLE_HINTS.has(k))
    const hasNonEdibleKw = textKw.some(k => NON_EDIBLE_HINTS.has(k))

    let isEdible = false
    let score = 0.5

    if (edibleLikelyFromImage) { isEdible = true; score = 0.8 }
    if (hasEdibleKw) { isEdible = true; score = Math.max(score, 0.75) }
    if (hasNonEdibleKw) { isEdible = false; score = Math.max(score, 0.75) }

    return { isEdible, confidence: Math.round(clamp(score, 0, 1) * 100) }
  }

  async function classifyText (text) {
    const { textClassifier } = await ensurePipelines()
    const candidate_labels = TEXT_LABELS
    const out = await textClassifier(text || 'unknown item', candidate_labels, { multi_label: true })
    // `out` can be array or object depending on version
    const labels = out.labels || (out[0] && out[0].labels) || []
    const scores = out.scores || (out[0] && out[0].scores) || []
    // top label
    let topIdx = 0, topScore = -1
    for (let i = 0; i < labels.length; i++) {
      if (scores[i] > topScore) { topScore = scores[i]; topIdx = i }
    }
    return { labels, scores, top: { label: labels[topIdx] || 'unknown', score: scores[topIdx] || 0 } }
  }

  async function classifyImage (imageInput) {
    const { imageClassifier } = await ensurePipelines()
    // imageInput can be Blob/File/HTMLImageElement/ImageBitmap/URL
    const out = await imageClassifier(imageInput, { topk: 5 })
    // normalize to {label, score}
    const top = Array.isArray(out) && out.length ? out[0] : { label: 'unknown', score: 0 }
    return { top, all: out }
  }

  function normalizeImageInput (image) {
    // accept File/Blob/DataURL/HTMLImageElement/Canvas/ImageBitmap
    if (!image) return null
    if (typeof image === 'string') return image // assume URL/dataURL
    if (image instanceof Blob) return image
    if (image instanceof HTMLImageElement) return image
    if (image instanceof HTMLCanvasElement) return image
    if ('transferToImageBitmap' in image) return image.transferToImageBitmap()
    // Unknown type — return as-is; transformers may still handle it
    return image
  }

  /**
   * Main entry — maintains the same signature the app expects.
   * @param {{ text?: string, image?: any, imageName?: string, location?: string }} input
   * @returns {Promise<object>} result
   */
  EcoAI.analyze = async function analyze (input = {}) {
    try {
      const { text = '', image = null, imageName = '', location = '' } = input
      const textKw = keywords(text + ' ' + imageName)

      let textInfo = null
      let imageInfo = null

      if (text && text.trim().length > 0) {
        textInfo = await classifyText(text)
      }
      if (image) {
        const img = normalizeImageInput(image)
        imageInfo = await classifyImage(img)
      }

      const textTop = textInfo ? textInfo.top : null
      const imageTop = imageInfo ? imageInfo.top : null

      // Decide edibility
      const edible = edibleDecision({ textTop, imageTop, textKw })

      // Pick a semantic label to drive eco/nutrition
      let drivingLabel = (textTop && textTop.label) || (imageTop && imageTop.label) || 'unknown'
      // Map arbitrary image labels to a coarse bucket
      const coarse = (() => {
        const l = (drivingLabel || '').toLowerCase()
        if (/(banana|apple|orange|fruit)/.test(l)) return 'fruit'
        if (/(tomato|potato|vegetable|cabbage|carrot)/.test(l)) return 'vegetable'
        if (/(milk|yogurt|cheese|dairy)/.test(l)) return 'dairy'
        if (/(meat|beef|chicken|pork|fish|tuna)/.test(l)) return 'meat'
        if (/(drink|beverage|bottle|can|juice|soda|coffee|tea|water)/.test(l)) return 'drink'
        if (/(snack|chips|chocolate|cookie|biscuit|candy|bar)/.test(l)) return 'snack'
        if (/(detergent|cleaner|soap|shampoo|bleach)/.test(l)) return 'cleaning product'
        if (/(phone|laptop|camera|electronic|remote|battery)/.test(l)) return 'electronic device'
        if (/(lipstick|cream|cosmetic|makeup|lotion)/.test(l)) return 'cosmetic'
        if (/(shirt|jacket|shoe|clothing|t-shirt)/.test(l)) return 'clothing'
        return (textTop && TEXT_LABELS.includes(textTop.label)) ? textTop.label : 'food product'
      })()

      const nutrition = edible.isEdible ? guessNutrition(coarse) : null
      const eco = ecoAssessmentFromLabel(coarse)
      const alternatives = alternativesFromLabel(coarse)
      const tips = tipsFromContext(textKw)

      const overallConfidence = Math.round(clamp(
        ( (textTop ? textTop.score : 0.5) + (imageTop ? imageTop.score : 0.5) + (edible.confidence/100) ) / 3,
        0, 1
      ) * 100)

      return {
        input: { text, imageName, location },
        model: {
          textTop: textTop || null,
          imageTop: imageTop || null
        },
        edible: {
          isEdible: !!edible.isEdible,
          confidence: edible.confidence,
          explain: edible.isEdible
            ? 'Inferred from model predictions and keywords indicating food/edible items.'
            : 'Inferred from model predictions or keywords indicating non-food items.'
        },
        nutrition,
        eco,
        alternatives,
        tips,
        overallConfidence
      }
    } catch (err) {
      console.error('EcoAI.analyze error:', err)
      throw err
    }
  }

  /**
   * Human-readable model explanation for the UI.
   * Uses top text/image labels and edible decision.
   */
  EcoAI.explain = function explain(res){
    try {
      const lines = [];
      if (res?.model?.textTop) {
        lines.push(`Text suggests: "${res.model.textTop.label}" (${Math.round(res.model.textTop.score*100)}%)`);
      }
      if (res?.model?.imageTop) {
        lines.push(`Image suggests: "${res.model.imageTop.label}" (${Math.round(res.model.imageTop.score*100)}%)`);
      }
      if (res?.edible) {
        lines.push(`Edible decision: ${res.edible.isEdible ? 'likely edible' : 'likely not edible'} (conf ${res.edible.confidence}%)`);
      }
      if (!lines.length) return 'No model signals available.';
      return lines.join(' • ');
    } catch(e){
      return 'Explanation unavailable.';
    }
  }

})()