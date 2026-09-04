/*
 * Pop Art IRL -- live pop-art / Ben-Day-dot rendering of the webcam.
 *
 *   1. Image-processing primitives (bilateral filter, k-means, Sobel edges)
 *   2. createRenderer -- the per-frame pop-art pipeline
 *   3. The web app -- camera, render loop, settings and share panels
 */

// ===========================================================================
// 1. Image-processing primitives
// ===========================================================================

function clamp(v, min, max) {
  return v < min ? min : v > max ? max : v;
}

// mulberry32 PRNG, seeded so the same options reproduce the same palette.
function createRng(seed) {
  let s = seed >>> 0;
  return function rng() {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function toGrayscale(imageData) {
  const { data, width, height } = imageData;
  const gray = new Float32Array(width * height);
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    gray[p] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  }
  return gray;
}

// Black/white points from the luminance histogram, ignoring the darkest and
// lightest clipFraction of pixels as outliers. Feeds applyLevels().
function computeAutoLevels(pixels, clipFraction = 0.02) {
  const histogram = new Uint32Array(256);
  const pixelCount = pixels.length / 4;
  for (let i = 0; i < pixels.length; i += 4) {
    const luminance = (0.299 * pixels[i] + 0.587 * pixels[i + 1] + 0.114 * pixels[i + 2]) | 0;
    histogram[luminance]++;
  }

  const clipCount = pixelCount * clipFraction;
  let cumulative = 0;
  let blackPoint = 0;
  for (let v = 0; v < 256; v++) {
    cumulative += histogram[v];
    if (cumulative > clipCount) {
      blackPoint = v;
      break;
    }
  }

  cumulative = 0;
  let whitePoint = 255;
  for (let v = 255; v >= 0; v--) {
    cumulative += histogram[v];
    if (cumulative > clipCount) {
      whitePoint = v;
      break;
    }
  }

  if (whitePoint <= blackPoint) return { blackPoint: 0, whitePoint: 255 };
  return { blackPoint, whitePoint };
}

// Rescales pixels so blackPoint maps to 0 and whitePoint to 255, one scale
// for all three channels (keeps hue).
function applyLevels(pixels, blackPoint, whitePoint) {
  const scale = 255 / Math.max(1, whitePoint - blackPoint);
  const out = new Uint8ClampedArray(pixels.length);
  for (let i = 0; i < pixels.length; i += 4) {
    out[i] = (pixels[i] - blackPoint) * scale;
    out[i + 1] = (pixels[i + 1] - blackPoint) * scale;
    out[i + 2] = (pixels[i + 2] - blackPoint) * scale;
    out[i + 3] = pixels[i + 3];
  }
  return out;
}

// Edge-preserving blur: neighbor weight falls off with both distance and
// color difference, so texture and noise smooth while real edges stay sharp.
function bilateralFilter(pixels, width, height, radius, colorSigma) {
  const size = width * height;
  const out = new Uint8ClampedArray(size * 4);

  if (radius <= 0) {
    out.set(pixels);
    return out;
  }

  const spatialSigma = Math.max(1, radius / 2);
  const kernelSize = radius * 2 + 1;
  const spatialWeights = new Float64Array(kernelSize * kernelSize);
  for (let dy = -radius; dy <= radius; dy++) {
    for (let dx = -radius; dx <= radius; dx++) {
      const d2 = dx * dx + dy * dy;
      spatialWeights[(dy + radius) * kernelSize + (dx + radius)] =
        Math.exp(-d2 / (2 * spatialSigma * spatialSigma));
    }
  }

  // Range weights indexed by summed squared RGB difference (0..3*255*255),
  // precomputed to avoid a Math.exp() per tap.
  const maxRangeSq = 3 * 255 * 255;
  const rangeWeights = new Float64Array(maxRangeSq + 1);
  const rangeDenom = 2 * colorSigma * colorSigma * 3;
  for (let d2 = 0; d2 <= maxRangeSq; d2++) {
    rangeWeights[d2] = Math.exp(-d2 / rangeDenom);
  }

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      const p = idx * 4;
      const cr = pixels[p];
      const cg = pixels[p + 1];
      const cb = pixels[p + 2];

      let sumR = 0;
      let sumG = 0;
      let sumB = 0;
      let sumW = 0;

      for (let dy = -radius; dy <= radius; dy++) {
        const yy = clamp(y + dy, 0, height - 1);
        for (let dx = -radius; dx <= radius; dx++) {
          const xx = clamp(x + dx, 0, width - 1);
          const np = (yy * width + xx) * 4;
          const nr = pixels[np];
          const ng = pixels[np + 1];
          const nb = pixels[np + 2];
          const dr = nr - cr;
          const dg = ng - cg;
          const db = nb - cb;
          const d2 = dr * dr + dg * dg + db * db;
          const w = spatialWeights[(dy + radius) * kernelSize + (dx + radius)] * rangeWeights[d2];
          sumR += nr * w;
          sumG += ng * w;
          sumB += nb * w;
          sumW += w;
        }
      }

      out[p] = sumR / sumW;
      out[p + 1] = sumG / sumW;
      out[p + 2] = sumB / sumW;
      out[p + 3] = 255;
    }
  }

  return out;
}

function samplePixels(pixels, width, height, sampleSize, rng) {
  const size = width * height;
  const n = Math.min(sampleSize, size);
  const samples = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const idx = Math.floor(rng() * size);
    const p = idx * 4;
    samples[i * 3] = pixels[p];
    samples[i * 3 + 1] = pixels[p + 1];
    samples[i * 3 + 2] = pixels[p + 2];
  }
  return samples;
}

// k-means++ seeding: each new centroid is picked with probability
// proportional to its squared distance from the nearest one chosen so far.
function kmeansPlusPlusInit(samples, k, rng) {
  const n = samples.length / 3;
  const centroids = [];

  let idx = Math.floor(rng() * n);
  centroids.push([samples[idx * 3], samples[idx * 3 + 1], samples[idx * 3 + 2]]);

  const nearestDistSq = new Float64Array(n).fill(Infinity);

  for (let c = 1; c < k; c++) {
    const [cr, cg, cb] = centroids[centroids.length - 1];
    for (let i = 0; i < n; i++) {
      const dr = samples[i * 3] - cr;
      const dg = samples[i * 3 + 1] - cg;
      const db = samples[i * 3 + 2] - cb;
      const d = dr * dr + dg * dg + db * db;
      if (d < nearestDistSq[i]) nearestDistSq[i] = d;
    }

    let sum = 0;
    for (let i = 0; i < n; i++) sum += nearestDistSq[i];

    if (sum === 0) {
      idx = Math.floor(rng() * n);
    } else {
      let r = rng() * sum;
      idx = 0;
      for (; idx < n - 1; idx++) {
        r -= nearestDistSq[idx];
        if (r <= 0) break;
      }
    }
    centroids.push([samples[idx * 3], samples[idx * 3 + 1], samples[idx * 3 + 2]]);
  }

  return centroids;
}

// Lloyd's algorithm.
function kmeansFit(samples, centroids, iterations) {
  const n = samples.length / 3;
  const k = centroids.length;
  const assignment = new Int32Array(n);

  for (let iter = 0; iter < iterations; iter++) {
    for (let i = 0; i < n; i++) {
      const r = samples[i * 3];
      const g = samples[i * 3 + 1];
      const b = samples[i * 3 + 2];
      let best = 0;
      let bestDist = Infinity;
      for (let c = 0; c < k; c++) {
        const dr = r - centroids[c][0];
        const dg = g - centroids[c][1];
        const db = b - centroids[c][2];
        const d = dr * dr + dg * dg + db * db;
        if (d < bestDist) {
          bestDist = d;
          best = c;
        }
      }
      assignment[i] = best;
    }

    const sums = Array.from({ length: k }, () => [0, 0, 0, 0]);
    for (let i = 0; i < n; i++) {
      const c = assignment[i];
      sums[c][0] += samples[i * 3];
      sums[c][1] += samples[i * 3 + 1];
      sums[c][2] += samples[i * 3 + 2];
      sums[c][3] += 1;
    }
    for (let c = 0; c < k; c++) {
      if (sums[c][3] > 0) {
        centroids[c] = [sums[c][0] / sums[c][3], sums[c][1] / sums[c][3], sums[c][2] / sums[c][3]];
      }
    }
  }

  return centroids;
}

// Repeatedly merges the closest pair of colors until shouldStop. Returns the
// merged list plus, per original index, its merged-group index (so a label
// map can be remapped with an O(pixels) lookup).
function agglomerateCentroids(centroids, shouldStop) {
  let current = centroids.map((color, i) => ({ color, weight: 1, sourceIndices: [i] }));

  while (current.length > 1) {
    let bestI = -1;
    let bestJ = -1;
    let bestDist = Infinity;

    for (let i = 0; i < current.length; i++) {
      for (let j = i + 1; j < current.length; j++) {
        const dr = current[i].color[0] - current[j].color[0];
        const dg = current[i].color[1] - current[j].color[1];
        const db = current[i].color[2] - current[j].color[2];
        const dist = Math.sqrt(dr * dr + dg * dg + db * db);
        if (dist < bestDist) {
          bestDist = dist;
          bestI = i;
          bestJ = j;
        }
      }
    }

    if (shouldStop(current.length, bestDist)) break;

    const a = current[bestI];
    const b = current[bestJ];
    const totalWeight = a.weight + b.weight;
    const merged = {
      color: [
        (a.color[0] * a.weight + b.color[0] * b.weight) / totalWeight,
        (a.color[1] * a.weight + b.color[1] * b.weight) / totalWeight,
        (a.color[2] * a.weight + b.color[2] * b.weight) / totalWeight,
      ],
      weight: totalWeight,
      sourceIndices: [...a.sourceIndices, ...b.sourceIndices],
    };
    current = current.filter((_, idx) => idx !== bestI && idx !== bestJ);
    current.push(merged);
  }

  const indexMap = new Array(centroids.length);
  current.forEach((group, groupIndex) => {
    for (const sourceIndex of group.sourceIndices) indexMap[sourceIndex] = groupIndex;
  });

  return { centroids: current.map((c) => c.color), indexMap };
}

function identityMerge(centroids) {
  return { centroids, indexMap: centroids.map((_, i) => i) };
}

// Merges palette colors closer together than minColorDistance.
function mergeCloseCentroids(centroids, minColorDistance) {
  if (minColorDistance <= 0) return identityMerge(centroids);
  return agglomerateCentroids(centroids, (_count, bestDist) => bestDist >= minColorDistance);
}

// Merges the palette down to targetCount colors regardless of distance.
function mergeCentroidsToCount(centroids, targetCount) {
  if (targetCount <= 0 || targetCount >= centroids.length) return identityMerge(centroids);
  return agglomerateCentroids(centroids, (count) => count <= targetCount);
}

function assignPixelsToCentroids(pixels, width, height, centroids) {
  const size = width * height;
  const k = centroids.length;
  const labels = new Int32Array(size);

  for (let idx = 0; idx < size; idx++) {
    const p = idx * 4;
    const r = pixels[p];
    const g = pixels[p + 1];
    const b = pixels[p + 2];
    let best = 0;
    let bestDist = Infinity;
    for (let c = 0; c < k; c++) {
      const dr = r - centroids[c][0];
      const dg = g - centroids[c][1];
      const db = b - centroids[c][2];
      const d = dr * dr + dg * dg + db * db;
      if (d < bestDist) {
        bestDist = d;
        best = c;
      }
    }
    labels[idx] = best;
  }

  return labels;
}

// Majority filter over the label map, to remove quantization speckle where
// two colors are near-equidistant.
function smoothLabels(labels, width, height, radius, numLabels) {
  if (radius <= 0) return labels;

  const out = new Int32Array(labels.length);
  const counts = new Int32Array(numLabels);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      counts.fill(0);
      for (let dy = -radius; dy <= radius; dy++) {
        const yy = clamp(y + dy, 0, height - 1);
        for (let dx = -radius; dx <= radius; dx++) {
          const xx = clamp(x + dx, 0, width - 1);
          counts[labels[yy * width + xx]]++;
        }
      }
      let best = 0;
      let bestCount = -1;
      for (let l = 0; l < numLabels; l++) {
        if (counts[l] > bestCount) {
          bestCount = counts[l];
          best = l;
        }
      }
      out[y * width + x] = best;
    }
  }

  return out;
}

// 4-connected components of the label map, with each component's label and area.
function computeConnectedComponents(labels, width, height) {
  const size = width * height;
  const componentIds = new Int32Array(size).fill(-1);
  const queue = new Int32Array(size);
  const componentLabel = [];
  const componentArea = [];
  let nextId = -1;

  for (let start = 0; start < size; start++) {
    if (componentIds[start] !== -1) continue;
    nextId++;
    const label = labels[start];
    let qHead = 0;
    let qTail = 0;
    queue[qTail++] = start;
    componentIds[start] = nextId;
    let area = 0;

    while (qHead < qTail) {
      const idx = queue[qHead++];
      area++;
      const x = idx % width;
      const y = (idx - x) / width;

      if (x > 0) {
        const n = idx - 1;
        if (componentIds[n] === -1 && labels[n] === label) { componentIds[n] = nextId; queue[qTail++] = n; }
      }
      if (x < width - 1) {
        const n = idx + 1;
        if (componentIds[n] === -1 && labels[n] === label) { componentIds[n] = nextId; queue[qTail++] = n; }
      }
      if (y > 0) {
        const n = idx - width;
        if (componentIds[n] === -1 && labels[n] === label) { componentIds[n] = nextId; queue[qTail++] = n; }
      }
      if (y < height - 1) {
        const n = idx + width;
        if (componentIds[n] === -1 && labels[n] === label) { componentIds[n] = nextId; queue[qTail++] = n; }
      }
    }

    componentLabel.push(label);
    componentArea.push(area);
  }

  return { componentIds, numComponents: nextId + 1, componentLabel, componentArea };
}

// Shared-border length between each pair of adjacent components.
function computeComponentAdjacency(componentIds, width, height) {
  const adjacency = new Map();
  const addEdge = (a, b) => {
    if (a === b) return;
    const key = a < b ? `${a}_${b}` : `${b}_${a}`;
    adjacency.set(key, (adjacency.get(key) || 0) + 1);
  };

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      const c = componentIds[idx];
      if (x < width - 1) {
        const n = componentIds[idx + 1];
        if (n !== c) addEdge(c, n);
      }
      if (y < height - 1) {
        const n = componentIds[idx + width];
        if (n !== c) addEdge(c, n);
      }
    }
  }

  return adjacency;
}

// Merges components smaller than minRegionArea into the neighbor they share
// the most border with.
function absorbSmallComponents(labels, width, height, minRegionArea, maxPasses) {
  if (minRegionArea <= 0) return labels;

  let current = labels;

  for (let pass = 0; pass < maxPasses; pass++) {
    const { componentIds, numComponents, componentLabel, componentArea } = computeConnectedComponents(
      current,
      width,
      height,
    );

    const smallComponents = [];
    for (let c = 0; c < numComponents; c++) {
      if (componentArea[c] < minRegionArea) smallComponents.push(c);
    }
    if (smallComponents.length === 0) break;

    const adjacency = computeComponentAdjacency(componentIds, width, height);
    const neighborsOf = new Map();
    for (const [key, weight] of adjacency) {
      const [a, b] = key.split('_').map(Number);
      if (!neighborsOf.has(a)) neighborsOf.set(a, []);
      if (!neighborsOf.has(b)) neighborsOf.set(b, []);
      neighborsOf.get(a).push([b, weight]);
      neighborsOf.get(b).push([a, weight]);
    }

    smallComponents.sort((a, b) => componentArea[a] - componentArea[b]);
    const relabel = new Map();

    for (const c of smallComponents) {
      const neighbors = neighborsOf.get(c);
      if (!neighbors || neighbors.length === 0) continue;

      let best = null;
      for (const [other, weight] of neighbors) {
        if (!best || weight > best.weight) best = { other, weight };
      }
      if (best) relabel.set(c, componentLabel[best.other]);
    }
    if (relabel.size === 0) break;

    const next = new Int32Array(current.length);
    for (let i = 0; i < current.length; i++) {
      const c = componentIds[i];
      next[i] = relabel.has(c) ? relabel.get(c) : current[i];
    }
    current = next;
  }

  return current;
}

function sobelGradients(gray, width, height) {
  const size = width * height;
  const gx = new Float32Array(size);
  const gy = new Float32Array(size);
  const mag = new Float32Array(size);
  const kx = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
  const ky = [-1, -2, -1, 0, 0, 0, 1, 2, 1];

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let sx = 0;
      let sy = 0;
      let k = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const xx = clamp(x + dx, 0, width - 1);
          const yy = clamp(y + dy, 0, height - 1);
          const v = gray[yy * width + xx];
          sx += v * kx[k];
          sy += v * ky[k];
          k++;
        }
      }
      const idx = y * width + x;
      gx[idx] = sx;
      gy[idx] = sy;
      mag[idx] = Math.sqrt(sx * sx + sy * sy);
    }
  }
  return { gx, gy, mag };
}

// Bilinear sample of mag. x1/y1 are clamped so an exactly-integer coordinate
// can't read past the array (undefined * 0 is NaN, not 0).
function sampleBilinear(mag, width, height, x, y) {
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(x0 + 1, width - 1);
  const y1 = Math.min(y0 + 1, height - 1);
  const fx = x - x0;
  const fy = y - y0;
  const v00 = mag[y0 * width + x0];
  const v10 = mag[y0 * width + x1];
  const v01 = mag[y1 * width + x0];
  const v11 = mag[y1 * width + x1];
  return v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy) + v01 * (1 - fx) * fy + v11 * fx * fy;
}

// Thins edges to one pixel wide by keeping only local maxima along the
// gradient direction.
function nonMaxSuppress(gx, gy, mag, width, height) {
  const out = new Float32Array(width * height);

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const idx = y * width + x;
      const m = mag[idx];
      if (m === 0) continue;

      const gxv = gx[idx];
      const gyv = gy[idx];
      const len = Math.sqrt(gxv * gxv + gyv * gyv) || 1;
      const dx = gxv / len;
      const dy = gyv / len;

      const n1 = sampleBilinear(mag, width, height, x + dx, y + dy);
      const n2 = sampleBilinear(mag, width, height, x - dx, y - dy);

      out[idx] = m >= n1 && m >= n2 ? m : 0;
    }
  }

  return out;
}

// Canny hysteresis: keep every pixel over `high`, plus pixels over `low`
// that connect through other low-passing pixels to a high one.
function hysteresisThreshold(mag, width, height, low, high) {
  const size = width * height;
  const out = new Uint8Array(size);
  const stack = new Int32Array(size);
  let top = 0;

  for (let i = 0; i < size; i++) {
    if (mag[i] >= high) {
      out[i] = 1;
      stack[top++] = i;
    }
  }

  while (top > 0) {
    const idx = stack[--top];
    const x = idx % width;
    const y = (idx - x) / width;

    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        if (dx === 0 && dy === 0) continue;
        const xx = x + dx;
        const yy = y + dy;
        if (xx < 0 || yy < 0 || xx >= width || yy >= height) continue;
        const n = yy * width + xx;
        if (!out[n] && mag[n] >= low) {
          out[n] = 1;
          stack[top++] = n;
        }
      }
    }
  }

  return out;
}

function dilateMask(mask, width, height, radius) {
  let current = mask;
  for (let r = 0; r < radius; r++) {
    const next = new Uint8Array(current.length);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = y * width + x;
        if (current[idx]) {
          next[idx] = 1;
          continue;
        }
        let hit = 0;
        for (let dy = -1; dy <= 1 && !hit; dy++) {
          for (let dx = -1; dx <= 1 && !hit; dx++) {
            const xx = x + dx;
            const yy = y + dy;
            if (xx < 0 || yy < 0 || xx >= width || yy >= height) continue;
            if (current[yy * width + xx]) hit = 1;
          }
        }
        next[idx] = hit;
      }
    }
    current = next;
  }
  return current;
}

// ===========================================================================
// 2. Pop art renderer
// ===========================================================================

const DEFAULTS = {
  width: 640,
  paletteSize: 12,
  bilateralRadius: 1,
  bilateralColorSigma: 40,
  minColorDistance: 30,
  minRegionArea: 50,
  regionMergePasses: 2,
  labelSmoothingRadius: 1,
  kmeansSampleSize: 4000,
  kmeansIterationsFirstFrame: 8,
  kmeansIterationsPerFrame: 3,
  renderPaletteSize: 5,
  patternTierFraction: 0.4,
  solidTierFraction: 0.4,
  stripeTierFraction: 0.5,
  patternSpacing: 15,
  patternMinCoverageFactor: 0.18,
  patternMaxCoverageFactor: 0.38,
  paperColor: [255, 255, 255],
  popArtColors: true,
  hueSwitchMargin: 20,
  edgeThresholdHigh: 150,
  edgeThresholdLow: 45,
  edgeLineWidth: 2,
  edgeColor: [0, 0, 0],
  tipTaperLength: 8,
  paletteStability: 0.4,
  autoLevels: true,
  autoLevelsStability: 0.15,
  randomSeed: 1,
};

// Fixed bold palette; near-white and near-black tiers become flat white/black.
const POP_ART_HUES = [0, 30, 55, 130, 220, 285]; // red, orange, yellow, green, blue, purple
const POP_ART_SATURATION = 0.9;
const POP_ART_LIGHTNESS = 0.5;
const POP_ART_WHITE_LIGHTNESS = 0.85;
const POP_ART_BLACK_LIGHTNESS = 0.15;

function rgbToHsl(r, g, b) {
  r /= 255;
  g /= 255;
  b /= 255;
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const l = (max + min) / 2;
  if (max === min) return [0, 0, l];

  const d = max - min;
  const s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
  let h;
  if (max === r) h = (g - b) / d + (g < b ? 6 : 0);
  else if (max === g) h = (b - r) / d + 2;
  else h = (r - g) / d + 4;
  h *= 60;

  return [h, s, l];
}

function hue2rgb(p, q, t) {
  if (t < 0) t += 1;
  if (t > 1) t -= 1;
  if (t < 1 / 6) return p + (q - p) * 6 * t;
  if (t < 1 / 2) return q;
  if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
  return p;
}

function hslToRgb(h, s, l) {
  if (s === 0) {
    const v = Math.round(l * 255);
    return [v, v, v];
  }
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  const hk = h / 360;
  return [
    Math.round(hue2rgb(p, q, hk + 1 / 3) * 255),
    Math.round(hue2rgb(p, q, hk) * 255),
    Math.round(hue2rgb(p, q, hk - 1 / 3) * 255),
  ];
}

function hueAngleDistance(a, b) {
  const d = Math.abs(a - b);
  return Math.min(d, 360 - d);
}

function nearestPopHue(h) {
  let best = POP_ART_HUES[0];
  let bestDist = Infinity;
  for (const hue of POP_ART_HUES) {
    const d = hueAngleDistance(h, hue);
    if (d < bestDist) {
      bestDist = d;
      best = hue;
    }
  }
  return best;
}

function colorDistance(a, b) {
  const dr = a[0] - b[0];
  const dg = a[1] - b[1];
  const db = a[2] - b[2];
  return Math.sqrt(dr * dr + dg * dg + db * db);
}

// Snaps each color to the nearest bold hue, but keeps last frame's choice
// unless the fresh one is a better fit by more than hueSwitchMargin degrees
// -- otherwise lighting drift flips a whole region between two bold colors
// every frame. Matching is by color, not screen position.
function createHueResolver() {
  let previousCommitments = [];

  function resolve(rawColor, hueSwitchMargin, nextCommitments) {
    const [h, , l] = rgbToHsl(rawColor[0], rawColor[1], rawColor[2]);
    if (l > POP_ART_WHITE_LIGHTNESS) return [255, 255, 255];
    if (l < POP_ART_BLACK_LIGHTNESS) return [0, 0, 0];

    const freshHue = nearestPopHue(h);

    let match = null;
    let bestDist = Infinity;
    for (const commitment of previousCommitments) {
      const d = colorDistance(rawColor, commitment.color);
      if (d < bestDist) {
        bestDist = d;
        match = commitment;
      }
    }

    let hue = freshHue;
    if (match && bestDist < 60 && hueAngleDistance(h, match.hue) - hueAngleDistance(h, freshHue) <= hueSwitchMargin) {
      hue = match.hue;
    }

    nextCommitments.push({ color: rawColor, hue });
    return hslToRgb(hue, POP_ART_SATURATION, POP_ART_LIGHTNESS);
  }

  return {
    resolveAll(rawColors, hueSwitchMargin) {
      const nextCommitments = [];
      const result = rawColors.map((c) => resolve(c, hueSwitchMargin, nextCommitments));
      previousCommitments = nextCommitments;
      return result;
    },
  };
}

// Per-pixel fade (0-1) for outline pixels near a stroke's loose end. Tips are
// thin-mask pixels with at most one neighbor; the fade is a BFS out from each
// tip over the dilated mask, so it follows the stroke rather than straight
// lines.
function computeTipTaper(thinMask, fullMask, width, height, taperLength) {
  const size = width * height;
  const taper = new Float32Array(size).fill(1);
  if (taperLength <= 0) return taper;

  const isTip = new Uint8Array(size);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      if (!thinMask[idx]) continue;
      let neighbors = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          const xx = x + dx;
          const yy = y + dy;
          if (xx < 0 || yy < 0 || xx >= width || yy >= height) continue;
          if (thinMask[yy * width + xx]) neighbors++;
        }
      }
      if (neighbors <= 1) isTip[idx] = 1;
    }
  }

  const distance = new Int32Array(size).fill(-1);
  const queue = new Int32Array(size);
  let queueTail = 0;
  for (let i = 0; i < size; i++) {
    if (isTip[i]) {
      distance[i] = 0;
      queue[queueTail++] = i;
    }
  }

  let queueHead = 0;
  while (queueHead < queueTail) {
    const idx = queue[queueHead++];
    const d = distance[idx];
    if (d >= taperLength) continue;
    const x = idx % width;
    const y = (idx - x) / width;
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        if (dx === 0 && dy === 0) continue;
        const xx = x + dx;
        const yy = y + dy;
        if (xx < 0 || yy < 0 || xx >= width || yy >= height) continue;
        const n = yy * width + xx;
        if (!fullMask[n] || distance[n] !== -1) continue;
        distance[n] = d + 1;
        queue[queueTail++] = n;
      }
    }
  }

  for (let i = 0; i < size; i++) {
    if (!fullMask[i] || distance[i] === -1) continue;
    const s = Math.min(1, distance[i] / taperLength);
    taper[i] = s * s * (3 - 2 * s);
  }

  return taper;
}

/*
 * Stateful pop-art renderer. Call processFrame(source) once per frame: the
 * palette and exposure persist across calls so the output doesn't reshuffle
 * frame to frame. updateOptions() applies settings live (changing paletteSize
 * forces a palette re-fit). width is fixed once the first frame sizes the
 * working canvas -- make a new renderer to change it. Options are in DEFAULTS.
 *
 * Returns { processFrame, updateOptions, width, height, frameCount, timings }.
 */
function createRenderer(options = {}) {
  const settings = { ...DEFAULTS, ...options };
  const rng = createRng(settings.randomSeed);

  let width = 0;
  let height = 0;
  let workingCanvas = null;
  let workingCtx = null;
  const outputCanvas = document.createElement('canvas');
  const outputCtx = outputCanvas.getContext('2d');

  let centroids = null;
  let levels = null;
  const hueResolver = createHueResolver();
  let frameCount = 0;
  const timings = {};

  // paletteSize can't warm-restart (kmeansFit assumes the count already
  // matches), so changing it drops the palette for a re-fit next frame.
  function updateOptions(newOptions) {
    if ('paletteSize' in newOptions && newOptions.paletteSize !== settings.paletteSize) {
      centroids = null;
    }
    Object.assign(settings, newOptions);
  }

  function ensureSized(sourceWidth, sourceHeight) {
    if (workingCanvas) return;
    const scale = settings.width / sourceWidth;
    width = Math.max(1, Math.round(settings.width));
    height = Math.max(1, Math.round(sourceHeight * scale));

    workingCanvas = document.createElement('canvas');
    workingCanvas.width = width;
    workingCanvas.height = height;
    workingCtx = workingCanvas.getContext('2d', { willReadFrequently: true });

    outputCanvas.width = width;
    outputCanvas.height = height;
  }

  function processFrame(source) {
    const {
      paletteSize,
      bilateralRadius,
      bilateralColorSigma,
      minColorDistance,
      minRegionArea,
      regionMergePasses,
      labelSmoothingRadius,
      kmeansSampleSize,
      kmeansIterationsFirstFrame,
      kmeansIterationsPerFrame,
      renderPaletteSize,
      patternTierFraction,
      solidTierFraction,
      stripeTierFraction,
      patternSpacing,
      patternMinCoverageFactor,
      patternMaxCoverageFactor,
      paperColor,
      popArtColors,
      hueSwitchMargin,
      edgeThresholdHigh,
      edgeThresholdLow,
      edgeLineWidth,
      edgeColor,
      tipTaperLength,
      paletteStability,
      autoLevels,
      autoLevelsStability,
    } = settings;

    const sourceWidth = source.videoWidth || source.width;
    const sourceHeight = source.videoHeight || source.height;
    if (!sourceWidth || !sourceHeight) return outputCanvas;

    const t0 = performance.now();
    ensureSized(sourceWidth, sourceHeight);

    workingCtx.drawImage(source, 0, 0, width, height);
    const pixels = workingCtx.getImageData(0, 0, width, height).data;
    const t1 = performance.now();

    // Rescale this frame's tonal range to fill 0-255, eased across frames so
    // exposure doesn't visibly pump.
    let normalized = pixels;
    if (autoLevels) {
      const target = computeAutoLevels(pixels);
      if (!levels) levels = target;
      levels = {
        blackPoint: levels.blackPoint + (target.blackPoint - levels.blackPoint) * autoLevelsStability,
        whitePoint: levels.whitePoint + (target.whitePoint - levels.whitePoint) * autoLevelsStability,
      };
      normalized = applyLevels(pixels, levels.blackPoint, levels.whitePoint);
    }
    const t1b = performance.now();

    const smoothed = bilateralFilter(normalized, width, height, bilateralRadius, bilateralColorSigma);
    const t2 = performance.now();

    if (!centroids) {
      const samples = samplePixels(smoothed, width, height, Math.max(kmeansSampleSize, 8000), rng);
      const initial = kmeansPlusPlusInit(samples, paletteSize, rng);
      centroids = kmeansFit(samples, initial, kmeansIterationsFirstFrame);
    } else {
      // Blend toward the fresh fit instead of replacing it, to damp the
      // per-frame k-means jitter that would otherwise flicker hue buckets.
      const samples = samplePixels(smoothed, width, height, kmeansSampleSize, rng);
      const previous = centroids.map((c) => c.slice());
      const fitted = kmeansFit(samples, centroids, kmeansIterationsPerFrame);
      centroids = fitted.map((c, i) => [
        previous[i][0] + (c[0] - previous[i][0]) * paletteStability,
        previous[i][1] + (c[1] - previous[i][1]) * paletteStability,
        previous[i][2] + (c[2] - previous[i][2]) * paletteStability,
      ]);
    }
    const t3 = performance.now();

    const { centroids: fillCentroids } = mergeCloseCentroids(centroids, minColorDistance);
    const rawLabels = assignPixelsToCentroids(smoothed, width, height, fillCentroids);
    const speckleCleaned = smoothLabels(rawLabels, width, height, labelSmoothingRadius, fillCentroids.length);
    const labels = absorbSmallComponents(speckleCleaned, width, height, minRegionArea, regionMergePasses);
    const t4 = performance.now();

    // Coarsen only the colors drawn, not the region shapes: renderGroupOf
    // maps a fine label to one of these bold groups.
    const { centroids: renderGroups, indexMap: renderGroupOf } = mergeCentroidsToCount(
      fillCentroids,
      renderPaletteSize,
    );

    // Darkest patternTierFraction of the groups aren't flat-filled; the
    // darkest solidTierFraction of those are solid ink and the rest get a
    // halftone pattern. patternRankOf ranks the patterned groups (0 =
    // lightest) so ink coverage can scale with darkness.
    const groupCount = renderGroups.length;
    const luminance = renderGroups.map((c) => 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]);
    const lightestFirst = luminance.map((_, i) => i).sort((a, b) => luminance[b] - luminance[a]);
    const patternTierCount = Math.round(groupCount * patternTierFraction);
    const solidTierCount = Math.round(patternTierCount * solidTierFraction);
    const patternedTierCount = patternTierCount - solidTierCount;
    const isPatternTier = new Array(groupCount).fill(false);
    const patternRankOf = new Array(groupCount).fill(-1);
    for (let i = 0; i < patternTierCount; i++) {
      const group = lightestFirst[groupCount - patternTierCount + i];
      if (i >= patternedTierCount) continue; // darkest tiers stay flat solid ink
      isPatternTier[group] = true;
      patternRankOf[group] = i;
    }

    // Stripe vs. dot is split on the fine labels, not the render groups:
    // with only 1-2 patterned groups, stripeTierFraction could only ever be
    // all-or-nothing.
    const fineLuminance = fillCentroids.map((c) => 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]);
    const patternedFineLabels = [];
    for (let label = 0; label < fillCentroids.length; label++) {
      if (isPatternTier[renderGroupOf[label]]) patternedFineLabels.push(label);
    }
    patternedFineLabels.sort((a, b) => fineLuminance[b] - fineLuminance[a]);
    const stripeFineCount = Math.round(patternedFineLabels.length * stripeTierFraction);
    const isStripeLabel = new Array(fillCentroids.length).fill(false);
    for (let i = 0; i < stripeFineCount; i++) {
      const label = patternedFineLabels[patternedFineLabels.length - stripeFineCount + i];
      isStripeLabel[label] = true;
    }

    // popArtColors swaps in the bold hues; the tiering above still used the
    // real luminance.
    const renderColors = popArtColors ? hueResolver.resolveAll(renderGroups, hueSwitchMargin) : renderGroups;
    const t4b = performance.now();

    function coverageForRank(rank) {
      const t = patternedTierCount > 1 ? rank / (patternedTierCount - 1) : 1;
      return patternMinCoverageFactor + (patternMaxCoverageFactor - patternMinCoverageFactor) * t;
    }

    // Everything below is recomputed fresh each frame with no cross-frame
    // smoothing: smoothing hides boundary flicker but smears the whole image
    // under camera motion. Flat groups take their color directly; pattern
    // groups take the paper color now (dots are painted below); stripe
    // groups get a diagonal duty-cycle test right here.
    const basePixels = new Uint8ClampedArray(width * height * 4);
    for (let idx = 0; idx < labels.length; idx++) {
      const label = labels[idx];
      const group = renderGroupOf[label];
      const p = idx * 4;
      let c;
      if (isStripeLabel[label]) {
        const coverage = coverageForRank(patternRankOf[group]);
        const x = idx % width;
        const y = (idx - x) / width;
        const phase = (x + y) % patternSpacing;
        c = phase < coverage * patternSpacing ? renderColors[group] : paperColor;
      } else if (isPatternTier[group]) {
        c = paperColor;
      } else {
        c = renderColors[group];
      }
      basePixels[p] = c[0];
      basePixels[p + 1] = c[1];
      basePixels[p + 2] = c[2];
      basePixels[p + 3] = 255;
    }
    const t4c = performance.now();

    // Halftone grid with offset rows. Painted pixel-by-pixel and clipped to
    // the same render group (not an arc() fill), so a dot never bleeds
    // across a boundary into a differently-textured neighbor.
    for (let gy = 0, row = 0; gy < height; gy += patternSpacing, row++) {
      const rowOffset = row % 2 === 1 ? patternSpacing / 2 : 0;
      for (let gx = rowOffset; gx < width; gx += patternSpacing) {
        const px = Math.min(width - 1, Math.round(gx));
        const py = Math.min(height - 1, Math.round(gy));
        const label = labels[py * width + px];
        const group = renderGroupOf[label];
        if (!isPatternTier[group] || isStripeLabel[label]) continue;

        const radius = patternSpacing * coverageForRank(patternRankOf[group]);
        const c = renderColors[group];

        const minX = Math.max(0, Math.floor(gx - radius));
        const maxX = Math.min(width - 1, Math.ceil(gx + radius));
        const minY = Math.max(0, Math.floor(gy - radius));
        const maxY = Math.min(height - 1, Math.ceil(gy + radius));
        const r2 = radius * radius;
        for (let y = minY; y <= maxY; y++) {
          const dy = y - gy;
          const rowBase = y * width;
          for (let x = minX; x <= maxX; x++) {
            const dx = x - gx;
            if (dx * dx + dy * dy > r2) continue;
            const pixelIdx = rowBase + x;
            const pixelLabel = labels[pixelIdx];
            if (renderGroupOf[pixelLabel] !== group || isStripeLabel[pixelLabel]) continue;
            const p = pixelIdx * 4;
            basePixels[p] = c[0];
            basePixels[p + 1] = c[1];
            basePixels[p + 2] = c[2];
          }
        }
      }
    }
    outputCtx.putImageData(new ImageData(basePixels, width, height), 0, 0);
    const t6 = performance.now();

    // Reuses the already-smoothed buffer, so no second blur pass.
    const gray = toGrayscale({ data: smoothed, width, height });
    const { gx: sgx, gy: sgy, mag } = sobelGradients(gray, width, height);
    const thinned = nonMaxSuppress(sgx, sgy, mag, width, height);
    const rawEdges = hysteresisThreshold(thinned, width, height, edgeThresholdLow, edgeThresholdHigh);
    const rawOutline = edgeLineWidth > 0 ? dilateMask(rawEdges, width, height, edgeLineWidth) : rawEdges;
    const t7 = performance.now();

    const taper = computeTipTaper(rawEdges, rawOutline, width, height, tipTaperLength);
    const t7b = performance.now();

    // taper is the blend alpha, so edges anti-alias and fade at loose ends.
    const imgData = outputCtx.getImageData(0, 0, width, height);
    const data = imgData.data;
    for (let i = 0; i < rawOutline.length; i++) {
      if (!rawOutline[i]) continue;
      const alpha = taper[i];
      if (alpha < 0.02) continue;
      const p = i * 4;
      data[p] += (edgeColor[0] - data[p]) * alpha;
      data[p + 1] += (edgeColor[1] - data[p + 1]) * alpha;
      data[p + 2] += (edgeColor[2] - data[p + 2]) * alpha;
    }
    outputCtx.putImageData(imgData, 0, 0);
    const t9 = performance.now();

    frameCount++;
    timings.capture = t1 - t0;
    timings.autoLevels = t1b - t1;
    timings.bilateral = t2 - t1b;
    timings.kmeans = t3 - t2;
    timings.labels = t4 - t3;
    timings.tiering = t4b - t4;
    timings.baseFill = t4c - t4b;
    timings.dots = t6 - t4c;
    timings.edges = t7 - t6;
    timings.taper = t7b - t7;
    timings.composite = t9 - t7b;
    timings.total = t9 - t0;

    return outputCanvas;
  }

  return {
    processFrame,
    updateOptions,
    get width() {
      return width;
    },
    get height() {
      return height;
    },
    get frameCount() {
      return frameCount;
    },
    get timings() {
      return timings;
    },
  };
}

// ===========================================================================
// 3. Web app
// ===========================================================================

const LONG_PRESS_MS = 500;
const DOUBLE_TAP_MS = 400;

const TITLE_COLORS = ["#ff2d55", "#1e90ff", "#ffd400", "#00a800", "#a800a8", "#0a0a0a"];
const TITLE_SWAP_MS = 500;
const DEFAULT_PROMPT = "Click anywhere to begin";

const SHARE_URL = "https://mrericsir.github.io/popartirl/";
const QR_MODULE_PX = 6;
const QR_QUIET_ZONE_MODULES = 4;
const QR_INK = "#000000";
const QR_PAPER = "#ffffff";

const IDEAL_VIDEO_SIZE = { width: { ideal: 640 }, height: { ideal: 480 } };

// Slider ids whose control shows a 0-100 percentage for a 0-1 option.
const PERCENT_OPTIONS = new Set([
  "patternTierFraction",
  "solidTierFraction",
  "stripeTierFraction",
  "patternMaxCoverageFactor",
]);

const SLIDER_IDS = [
  "width",
  "paletteSize",
  "bilateralRadius",
  "bilateralColorSigma",
  "minColorDistance",
  "minRegionArea",
  "renderPaletteSize",
  "patternTierFraction",
  "solidTierFraction",
  "stripeTierFraction",
  "patternSpacing",
  "patternMaxCoverageFactor",
  "edgeThresholdHigh",
  "edgeThresholdLow",
  "edgeLineWidth",
  "tipTaperLength",
  "paletteStability",
  "hueSwitchMargin",
];

const CHECKBOX_IDS = ["popArtColors", "autoLevels"];

const video = document.getElementById("webcam");
const outputCanvas = document.getElementById("output");
const outputCtx = outputCanvas.getContext("2d");
const overlay = document.getElementById("overlay");
const brand = document.getElementById("brand");
const statusMessage = document.getElementById("statusMessage");
const statsPanel = document.getElementById("stats");

const toggleButton = document.getElementById("toggleButton");
const resetButton = document.getElementById("resetButton");
const controlPanel = document.getElementById("controlPanel");
const closePanelButton = document.getElementById("closePanelButton");
const qrPanel = document.getElementById("qrPanel");
const closeQrPanelButton = document.getElementById("closeQrPanelButton");
const qrCanvas = document.getElementById("qrCanvas");
const openSettingsButton = document.getElementById("openSettingsButton");
const mirrorInput = document.getElementById("mirror");

// The renderer takes a plain CanvasImageSource, so each webcam frame is
// copied here first (flipped when the mirror checkbox is set).
const sourceCanvas = document.createElement("canvas");
const sourceCtx = sourceCanvas.getContext("2d");

let renderer = null;
let running = false;
let rafHandle = null;
let stream = null;
let cameraStarting = false;
let switchingCamera = false;
let frameTimestamps = [];

let longPressTimer = null;
let longPressFired = false;
let lastTapTime = 0;
let videoDevices = [];
let currentDeviceIndex = 0;
let titleColorIndex = 0;

//
// Settings panel
//

function sliderValue(id) {
  const raw = Number(document.getElementById(id).value);
  return PERCENT_OPTIONS.has(id) ? raw / 100 : raw;
}

function readOptions() {
  const options = {};
  for (const id of SLIDER_IDS) options[id] = sliderValue(id);
  for (const id of CHECKBOX_IDS) options[id] = document.getElementById(id).checked;
  return options;
}

function applyDefaults() {
  for (const id of SLIDER_IDS) {
    const value = DEFAULTS[id];
    document.getElementById(id).value = PERCENT_OPTIONS.has(id) ? value * 100 : value;
  }
  for (const id of CHECKBOX_IDS) document.getElementById(id).checked = DEFAULTS[id];
  mirrorInput.checked = true;
}

function syncSliderLabels() {
  for (const id of SLIDER_IDS) {
    const suffix = PERCENT_OPTIONS.has(id) ? "%" : "";
    document.getElementById(id + "Value").textContent = document.getElementById(id).value + suffix;
  }
}

// width needs a fresh renderer (the working canvas is sized once); every
// other option applies live.
function onSliderChange(id) {
  if (!running) return;
  if (id === "width") recreateRenderer();
  else renderer.updateOptions(readOptions());
}

//
// Splash screen
//

// Button under the title is both the click prompt and the status readout.
function setStatus(message, isError = false) {
  statusMessage.textContent = message || DEFAULT_PROMPT;
  statusMessage.classList.toggle("error", isError && Boolean(message));
}

function showOverlay(show) {
  overlay.classList.toggle("hidden", !show);
}

function swapTitleColor() {
  if (overlay.classList.contains("hidden")) return;
  brand.style.color = TITLE_COLORS[titleColorIndex];
  titleColorIndex = (titleColorIndex + 1) % TITLE_COLORS.length;
}

//
// Render loop
//

function recreateRenderer() {
  renderer = createRenderer(readOptions());
  frameTimestamps = [];
}

function copyWebcamFrame() {
  const w = video.videoWidth;
  const h = video.videoHeight;
  if (sourceCanvas.width !== w || sourceCanvas.height !== h) {
    sourceCanvas.width = w;
    sourceCanvas.height = h;
  }

  sourceCtx.save();
  if (mirrorInput.checked) {
    sourceCtx.translate(w, 0);
    sourceCtx.scale(-1, 1);
  }
  sourceCtx.drawImage(video, 0, 0, w, h);
  sourceCtx.restore();
}

function drawResult(result) {
  if (outputCanvas.width !== result.width || outputCanvas.height !== result.height) {
    outputCanvas.width = result.width;
    outputCanvas.height = result.height;
  }
  outputCtx.drawImage(result, 0, 0);
}

function updateStats() {
  const now = performance.now();
  frameTimestamps.push(now);
  while (now - frameTimestamps[0] > 1000) frameTimestamps.shift();

  const timings = renderer.timings;
  if (timings.total == null) return;
  statsPanel.textContent =
    `fps (last 1s):   ${frameTimestamps.length}\n` +
    `working size:    ${renderer.width}x${renderer.height}\n` +
    `frame:           ${renderer.frameCount}\n` +
    `total per frame: ${timings.total.toFixed(1)} ms`;
}

function frameTick() {
  if (!running) return;
  if (video.videoWidth && video.videoHeight) {
    copyWebcamFrame();
    drawResult(renderer.processFrame(sourceCanvas));
    updateStats();
  }
  rafHandle = requestAnimationFrame(frameTick);
}

function startLoop() {
  stopLoop();
  running = true;
  recreateRenderer();
  rafHandle = requestAnimationFrame(frameTick);
}

function stopLoop() {
  running = false;
  if (rafHandle != null) {
    cancelAnimationFrame(rafHandle);
    rafHandle = null;
  }
}

//
// Camera
//

function getStream(videoConstraints) {
  return navigator.mediaDevices.getUserMedia({
    video: { ...IDEAL_VIDEO_SIZE, ...videoConstraints },
    audio: false,
  });
}

function stopStream() {
  if (!stream) return;
  stream.getTracks().forEach((track) => track.stop());
  stream = null;
}

// Show a new stream in the <video>, defaulting the mirror on unless it is a
// rear camera.
async function attachStream(newStream) {
  stream = newStream;
  video.srcObject = newStream;
  await video.play();
  const facingMode = newStream.getVideoTracks()[0]?.getSettings().facingMode;
  mirrorInput.checked = facingMode !== "environment";
}

async function refreshVideoDevices() {
  if (!navigator.mediaDevices?.enumerateDevices) return;
  const devices = await navigator.mediaDevices.enumerateDevices();
  videoDevices = devices.filter((d) => d.kind === "videoinput");
}

async function startCamera() {
  if (stream || cameraStarting) return;

  cameraStarting = true;
  setStatus("Requesting camera...");
  try {
    await attachStream(await getStream({ facingMode: "user" }));
  } catch (err) {
    stopStream();
    setStatus(`Could not access the camera: ${err.message}`, true);
    cameraStarting = false;
    return;
  }

  await refreshVideoDevices();
  const activeId = stream.getVideoTracks()[0]?.getSettings().deviceId;
  currentDeviceIndex = Math.max(0, videoDevices.findIndex((d) => d.deviceId === activeId));

  cameraStarting = false;
  setStatus("");
  showOverlay(false);
  toggleButton.disabled = false;
  startLoop();
}

function stopCamera() {
  stopStream();
  video.srcObject = null;
  stopLoop();
  outputCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
  statsPanel.textContent = "Camera off.";
  setStatus("");
  toggleButton.disabled = true;
  showOverlay(true);
}

async function switchCamera() {
  if (!stream || switchingCamera) return;

  // Phones expose front/back through facingMode rather than stable deviceIds,
  // so toggle facingMode when we can and cycle deviceIds otherwise.
  const facingMode = stream.getVideoTracks()[0]?.getSettings().facingMode;
  const canToggleFacingMode = facingMode === "user" || facingMode === "environment";

  if (!canToggleFacingMode && videoDevices.length < 2) {
    setStatus("Only one camera found");
    setTimeout(() => setStatus(""), 2000);
    return;
  }

  switchingCamera = true;
  const previousStream = stream;

  let nextIndex = currentDeviceIndex;
  let videoConstraints;
  if (canToggleFacingMode) {
    videoConstraints = { facingMode: facingMode === "user" ? "environment" : "user" };
  } else {
    nextIndex = (currentDeviceIndex + 1) % videoDevices.length;
    videoConstraints = { deviceId: { exact: videoDevices[nextIndex].deviceId } };
  }

  try {
    await attachStream(await getStream(videoConstraints));
  } catch (err) {
    setStatus(`Could not switch camera: ${err.message}`, true);
    switchingCamera = false;
    return;
  }

  previousStream.getTracks().forEach((track) => track.stop());
  currentDeviceIndex = nextIndex;
  recreateRenderer(); // new source dimensions need a new working canvas
  switchingCamera = false;
}

//
// Panels
//

function setPanel(panel, open) {
  panel.classList.toggle("active", open);
  const anyOpen = controlPanel.classList.contains("active") || qrPanel.classList.contains("active");
  overlay.classList.toggle("dimmed", anyOpen);
}

let qrRendered = false;

function renderQrCode() {
  if (qrRendered) return;
  qrRendered = true;

  const qr = qrcode(0, "M");
  qr.addData(SHARE_URL);
  qr.make();

  const moduleCount = qr.getModuleCount();
  const size = (moduleCount + QR_QUIET_ZONE_MODULES * 2) * QR_MODULE_PX;
  const offset = QR_QUIET_ZONE_MODULES * QR_MODULE_PX;

  qrCanvas.width = size;
  qrCanvas.height = size;
  const ctx = qrCanvas.getContext("2d");
  ctx.fillStyle = QR_PAPER;
  ctx.fillRect(0, 0, size, size);
  ctx.fillStyle = QR_INK;
  for (let row = 0; row < moduleCount; row++) {
    for (let col = 0; col < moduleCount; col++) {
      if (qr.isDark(row, col)) {
        ctx.fillRect(offset + col * QR_MODULE_PX, offset + row * QR_MODULE_PX, QR_MODULE_PX, QR_MODULE_PX);
      }
    }
  }
}

function openQrPanel() {
  renderQrCode();
  setPanel(qrPanel, true);
}

//
// Wiring
//

overlay.addEventListener("click", startCamera);

toggleButton.addEventListener("click", () => (stream ? stopCamera() : startCamera()));

resetButton.addEventListener("click", () => {
  applyDefaults();
  syncSliderLabels();
  if (running) recreateRenderer();
});

// Long press outside a panel switches cameras.
document.addEventListener("pointerdown", (event) => {
  if (controlPanel.contains(event.target) || qrPanel.contains(event.target)) return;
  longPressFired = false;
  clearTimeout(longPressTimer);
  longPressTimer = setTimeout(() => {
    longPressFired = true;
    switchCamera();
  }, LONG_PRESS_MS);
});
document.addEventListener("pointercancel", () => clearTimeout(longPressTimer));

// Double-tap outside a panel closes an open panel, or opens the share panel.
document.addEventListener("pointerup", (event) => {
  clearTimeout(longPressTimer);
  if (longPressFired || controlPanel.contains(event.target) || qrPanel.contains(event.target)) return;

  const now = performance.now();
  const isDoubleTap = now - lastTapTime < DOUBLE_TAP_MS;
  lastTapTime = isDoubleTap ? 0 : now;
  if (!isDoubleTap) return;

  if (controlPanel.classList.contains("active")) setPanel(controlPanel, false);
  else if (qrPanel.classList.contains("active")) setPanel(qrPanel, false);
  else openQrPanel();
});

closePanelButton.addEventListener("click", () => setPanel(controlPanel, false));
closeQrPanelButton.addEventListener("click", () => setPanel(qrPanel, false));
openSettingsButton.addEventListener("click", () => {
  setPanel(qrPanel, false);
  setPanel(controlPanel, true);
});

document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") return;
  setPanel(controlPanel, false);
  setPanel(qrPanel, false);
});

if (navigator.mediaDevices) {
  navigator.mediaDevices.addEventListener("devicechange", refreshVideoDevices);
}

for (const id of SLIDER_IDS) {
  const input = document.getElementById(id);
  input.addEventListener("input", syncSliderLabels);
  input.addEventListener("change", () => {
    syncSliderLabels();
    onSliderChange(id);
  });
}
for (const id of CHECKBOX_IDS) {
  document.getElementById(id).addEventListener("change", () => onSliderChange(id));
}

window.addEventListener("beforeunload", stopStream);

//
// Startup
//

applyDefaults();
syncSliderLabels();
refreshVideoDevices();
swapTitleColor();
setInterval(swapTitleColor, TITLE_SWAP_MS);

if (!navigator.mediaDevices?.getUserMedia) {
  setStatus(
    "This browser does not support camera access (getUserMedia). Try a recent " +
      "Chrome, Firefox, or Safari over http://localhost or https.",
    true
  );
}
