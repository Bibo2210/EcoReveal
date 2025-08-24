/*
  ai.js — Real in-browser AI using Transformers.js (no API keys, free)
  - Text: zero-shot classification (Xenova/distilbert-base-uncased-mnli)
  - Image: image classification (Xenova/vit-base-patch16-224)
*/

;(() => {
  const EcoAI = {};
  window.EcoAI = EcoAI;

  let _pipelinesPromise = null;
  async function ensurePipelines() {
    if (_pipelinesPromise) return _pipelinesPromise;
    _pipelinesPromise = (async () => {
      // Load transformers.js at runtime
      if (!window.transformers) {
        await new Promise((resolve, reject) => {
          const s = document.createElement('script');
          s.src = 'https://cdn.jsdelivr.net/npm/@xenova/transformers/dist/transformers.min.js';
          s.async = true;
          s.onload = resolve;
          s.onerror = () => reject(new Error('Failed to load Transformers.js'));
          document.head.appendChild(s);
        });
      }
      const { pipeline, env } = window.transformers;
      env.allowLocalModels = false;
      env.backends.onnx.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/@xenova/transformers/dist/';

      const textClassifier  = await pipeline('zero-shot-classification', 'Xenova/distilbert-base-uncased-mnli');
      const imageClassifier = await pipeline('image-classification',       'Xenova/vit-base-patch16-224');
      return { textClassifier, imageClassifier };
    })();
    return _pipelinesPromise;
  }

  // …helpers (keywords, edible decision, nutrition & eco heuristics)…

  /**
   * Main entry – works for both quick text and scanner (text + image).
   * @param {{ text?: string, image?: File|Blob|HTMLImageElement|string, imageName?: string, location?: string }} input
   */
  EcoAI.analyze = async function analyze(input = {}) {
    const { textClassifier, imageClassifier } = await ensurePipelines();

    const text = (input.text || '').trim();
    const textOut = text ? await textClassifier(text, ['food product','drink','fruit','vegetable','snack','dairy','meat','cosmetic','cleaning product','electronic device','toy','clothing','medication','plastic packaging','battery']) : null;
    const textTop = Array.isArray(textOut?.labels) ? { label: textOut.labels[0], score: textOut.scores[0] } : null;

    let imageTop = null;
    if (input.image) {
      const imgRes = await imageClassifier(input.image, { topk: 5 });
      imageTop = Array.isArray(imgRes) && imgRes.length ? imgRes[0] : null;
    }

    // Decide edibility using model hints + keywords
    // (nutrition/eco/alternatives/tips derived from coarse label)
    // …building the same shape your UI expects…

    return {
      input: { text, imageName: input.imageName || '', location: input.location || '' },
      model: { textTop, imageTop },
      edible: { /* isEdible, confidence, explain */ },
      nutrition: /* or null */,
      eco: /* assessment */,
      alternatives: /* array */,
      tips: /* array */,
      overallConfidence: /* 0–100 */
    };
  };

  // Used by renderResult() in app.js
  EcoAI.explain = function explain(res){
    try {
      const lines = [];
      if (res?.model?.textTop)  lines.push(`Text suggests: "${res.model.textTop.label}" (${Math.round(res.model.textTop.score*100)}%)`);
      if (res?.model?.imageTop) lines.push(`Image suggests: "${res.model.imageTop.label}" (${Math.round(res.model.imageTop.score*100)}%)`);
      if (res?.edible)          lines.push(`Edible decision: ${res.edible.isEdible ? 'likely edible' : 'likely not edible'} (conf ${res.edible.confidence}%)`);
      return lines.length ? lines.join(' • ') : 'No model signals available.';
    } catch { return 'Explanation unavailable.'; }
  };
})();
