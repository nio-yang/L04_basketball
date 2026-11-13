import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.mjs";

const MODEL_ROOT = new URL("./assets/policy/student095_325/", import.meta.url);
const ORT_ROOT = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
ort.env.wasm.wasmPaths = ORT_ROOT;
ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;

let manifest;
let encoders = new Map();
let actor;
let hidden;
let chain = Promise.resolve();

function outputNamed(outputs, names, predicate, fallbackIndex) {
  const name = names.find(predicate) ?? names[fallbackIndex];
  if (!name || !outputs[name]) throw new Error(`Missing ONNX output: ${name ?? fallbackIndex}`);
  return outputs[name];
}

function buildEncoderInput(observation, block) {
  const segments = new Map(manifest.encoder_contract.input_segments.map((item) => [item.name, item]));
  const components = block.components.map((name) => {
    const segment = segments.get(name);
    if (!segment) throw new Error(`Unknown policy observation component: ${name}`);
    return { segment, values: observation.subarray(segment.start, segment.stop) };
  });
  const length = components.reduce((total, item) => total + item.values.length, 0);
  const values = new Float32Array(length);
  let offset = 0;
  for (const component of components) {
    values.set(component.values, offset);
    offset += component.values.length;
  }
  const dimensions = block.components.length === 1 && components[0].segment.shape.length > 1
    ? [1, ...components[0].segment.shape]
    : [1, length];
  return new ort.Tensor("float32", values, dimensions);
}

async function initialize() {
  const response = await fetch(new URL("manifest.json?v=2", MODEL_ROOT), { cache: "no-store" });
  if (!response.ok) throw new Error(`Could not load policy manifest (${response.status})`);
  manifest = await response.json();
  if (manifest.schema !== 1 || manifest.action_dim !== 27 || manifest.encoder_contract.input_dim !== 7898) {
    throw new Error("Unsupported Student 095 policy contract");
  }

  for (const block of manifest.encoder_contract.blocks) {
    const graph = manifest.graphs.find((item) => item.file.includes(block.name));
    if (!graph) throw new Error(`Missing ONNX graph for ${block.name}`);
    const session = await ort.InferenceSession.create(new URL(graph.file, MODEL_ROOT).href, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    encoders.set(block.name, { session, block });
  }
  const actorGraph = manifest.graphs.find((item) => item.file === "actor.onnx");
  if (!actorGraph) throw new Error("Missing GRU actor graph");
  actor = await ort.InferenceSession.create(new URL(actorGraph.file, MODEL_ROOT).href, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  hidden = new Float32Array(manifest.recurrent.layers * manifest.recurrent.hidden_size);
  const depthRangeM = manifest.student_observation.depth.normalized_range_m;
  if (!Array.isArray(depthRangeM) || depthRangeM.length !== 2 || depthRangeM[1] <= depthRangeM[0]) {
    throw new Error("Student 095 manifest is missing its depth normalization range");
  }
  self.postMessage({
    type: "ready",
    actionDim: manifest.action_dim,
    observationDim: manifest.observation_dim,
    policyLabel: "095 Student@325 · Teacher@22000 蒸馏",
    depthRangeM,
  });
}

async function infer(values) {
  const observation = values instanceof Float32Array ? values : Float32Array.from(values);
  if (observation.length !== manifest.encoder_contract.input_dim) {
    throw new Error(`Expected ${manifest.encoder_contract.input_dim} observation values, got ${observation.length}`);
  }

  const encoded = new Map();
  for (const [name, entry] of encoders) {
    const tensor = buildEncoderInput(observation, entry.block);
    const result = await entry.session.run({ [entry.session.inputNames[0]]: tensor });
    encoded.set(name, outputNamed(result, entry.session.outputNames, () => true, 0).data);
  }

  const latentSize = manifest.encoder_contract.latent_order
    .reduce((total, item) => total + item.shape.reduce((product, dimension) => product * dimension, 1), 0);
  const latent = new Float32Array(latentSize);
  const segments = new Map(manifest.encoder_contract.input_segments.map((item) => [item.name, item]));
  let offset = 0;
  for (const item of manifest.encoder_contract.latent_order) {
    const valuesForPart = item.kind === "encoder"
      ? encoded.get(item.name)
      : observation.subarray(segments.get(item.name).start, segments.get(item.name).stop);
    if (!valuesForPart) throw new Error(`Missing actor latent input: ${item.name}`);
    latent.set(valuesForPart, offset);
    offset += valuesForPart.length;
  }

  const actorInputs = {};
  for (const input of actor.inputNames) {
    if (input === "input") actorInputs[input] = new ort.Tensor("float32", latent, [1, latent.length]);
    else if (input.startsWith("hidden")) {
      actorInputs[input] = new ort.Tensor("float32", hidden, [manifest.recurrent.layers, 1, manifest.recurrent.hidden_size]);
    } else throw new Error(`Unexpected actor input: ${input}`);
  }
  const result = await actor.run(actorInputs);
  const action = outputNamed(result, actor.outputNames, (name) => name.startsWith("output"), 0).data;
  const nextHidden = outputNamed(result, actor.outputNames, (name) => name.startsWith("hidden"), 1).data;
  if (action.length !== manifest.action_dim || !action.every(Number.isFinite) || !nextHidden.every(Number.isFinite)) {
    throw new Error("Policy returned malformed or non-finite output");
  }
  hidden.set(nextHidden);
  const actionCopy = Float32Array.from(action);
  self.postMessage({ type: "action", action: actionCopy }, [actionCopy.buffer]);
}

async function handle(message) {
  if (message.type === "init") return initialize();
  if (!manifest) throw new Error("Policy worker is not initialized");
  if (message.type === "reset") {
    hidden.fill(0);
    self.postMessage({ type: "reset" });
  } else if (message.type === "step") {
    await infer(message.observation);
  } else {
    throw new Error(`Unknown policy command: ${message.type}`);
  }
}

self.onmessage = ({ data }) => {
  chain = chain.then(() => handle(data)).catch((error) => {
    self.postMessage({ type: "error", message: error.message || String(error) });
  });
};