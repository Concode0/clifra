<script setup lang="ts">
import { computed, ref } from "vue";

type Point = {
  x: number;
  y: number;
  z: number;
  rawZ: number;
};

type ScreenPoint = Point & {
  sx: number;
  sy: number;
  fill: string;
};

const samples = ref(240);
const distortion = ref(0.7);
const epochs = ref(80);
const rate = ref(0.04);
const alignment = ref(0);

function resetAlignment() {
  alignment.value = 0;
}

function alignManifold() {
  alignment.value = Math.min(0.985, 1 - Math.exp(-epochs.value * rate.value * 0.28));
}

function generate(flatten: number): Point[] {
  const count = Math.max(32, Math.floor(samples.value));
  const points: Point[] = [];

  for (let i = 0; i < count; i += 1) {
    const t = (i / (count - 1)) * Math.PI * 2;
    const x = Math.sin(t);
    const y = Math.sin(t) * Math.cos(t);
    const rawZ = distortion.value * x * y;
    points.push({ x, y, z: rawZ * (1 - flatten), rawZ });
  }

  return points;
}

function colorFor(value: number): string {
  const normalized = Math.max(0, Math.min(1, (value + distortion.value * 0.5) / Math.max(distortion.value, 0.001)));
  const hue = 216 - normalized * 156;
  return `hsl(${hue.toFixed(1)} 74% 52%)`;
}

function project(points: Point[]): ScreenPoint[] {
  const yaw = -0.62;
  const tilt = 0.68;
  const cosYaw = Math.cos(yaw);
  const sinYaw = Math.sin(yaw);

  return points.map((point) => {
    const rotatedX = point.x * cosYaw - point.y * sinYaw;
    const rotatedY = point.x * sinYaw + point.y * cosYaw;
    return {
      ...point,
      sx: 50 + rotatedX * 34,
      sy: 53 - (rotatedY * 18 * tilt + point.z * 42),
      fill: colorFor(point.rawZ),
    };
  });
}

function energy(points: Point[]): number {
  return points.reduce((total, point) => total + point.z * point.z, 0) / points.length;
}

const input = computed(() => generate(0));
const output = computed(() => generate(alignment.value));
const inputPoints = computed(() => project(input.value));
const outputPoints = computed(() => project(output.value));
const inputPath = computed(() => inputPoints.value.map((point) => `${point.sx},${point.sy}`).join(" "));
const outputPath = computed(() => outputPoints.value.map((point) => `${point.sx},${point.sy}`).join(" "));
const inputEnergy = computed(() => energy(input.value));
const outputEnergy = computed(() => energy(output.value));
</script>

<template>
  <section class="demo-shell" aria-label="clifra manifold alignment demo">
    <div class="demo-toolbar">
      <label class="demo-field">
        Samples <span>{{ samples }}</span>
        <input v-model.number="samples" type="range" min="64" max="640" step="16" @input="resetAlignment" />
      </label>
      <label class="demo-field">
        Distortion <span>{{ distortion.toFixed(2) }}</span>
        <input v-model.number="distortion" type="range" min="0" max="2" step="0.05" @input="resetAlignment" />
      </label>
      <label class="demo-field">
        Epochs <span>{{ epochs }}</span>
        <input v-model.number="epochs" type="range" min="10" max="240" step="10" />
      </label>
      <label class="demo-field">
        Rate <span>{{ rate.toFixed(3) }}</span>
        <input v-model.number="rate" type="range" min="0.005" max="0.08" step="0.005" />
      </label>
      <button class="demo-button" type="button" @click="alignManifold">Align</button>
      <button class="demo-button secondary" type="button" @click="resetAlignment">Reset</button>
    </div>

    <div class="demo-stage">
      <article class="demo-panel">
        <header>
          <h3>Input manifold</h3>
          <output>z energy {{ inputEnergy.toFixed(5) }}</output>
        </header>
        <svg class="demo-plot" viewBox="0 0 100 75" role="img" aria-label="Distorted manifold point cloud">
          <polyline :points="inputPath" fill="none" stroke="currentColor" stroke-opacity="0.18" stroke-width="0.5" />
          <circle
            v-for="(point, index) in inputPoints"
            :key="`input-${index}`"
            :cx="point.sx"
            :cy="point.sy"
            r="0.8"
            :fill="point.fill"
            opacity="0.86"
          />
        </svg>
      </article>

      <article class="demo-panel">
        <header>
          <h3>Aligned manifold</h3>
          <output>z energy {{ outputEnergy.toFixed(5) }}</output>
        </header>
        <svg class="demo-plot" viewBox="0 0 100 75" role="img" aria-label="Aligned manifold point cloud">
          <polyline :points="outputPath" fill="none" stroke="currentColor" stroke-opacity="0.18" stroke-width="0.5" />
          <circle
            v-for="(point, index) in outputPoints"
            :key="`output-${index}`"
            :cx="point.sx"
            :cy="point.sy"
            r="0.8"
            :fill="point.fill"
            opacity="0.86"
          />
        </svg>
      </article>
    </div>
  </section>
</template>
