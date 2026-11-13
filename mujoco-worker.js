import loadMujoco from "https://cdn.jsdelivr.net/npm/@mujoco/mujoco@3.17.0/mujoco.js";

const BUNDLE_ROOT = new URL("./assets/sim_bundle/l04_v61/", import.meta.url);
const MODEL_PATH = "/l04/scene.mjb";
let mujoco;
let model;
let data;
let vfs;
let manifest;
let cameraBodyId = -1;
let wristBodyIds = [-1, -1];
let ballGeomIds = [];
let robotBodyIds = new Set();
let groundGeomIds = new Set();
let contactForceBuffer = null;
let lastContactSummary = null;
let catchMetrics = null;
let ballLaunched = false;
let previousPalmPositions = null;
let previousPalmTime = null;
let delayedTargets = [];
let commandQueue = Promise.resolve();

function clamp(value, lower, upper) {
  return Math.max(lower, Math.min(upper, value));
}

function identityBallPose(qposAddress, position) {
  data.qpos.set([position[0], position[1], position[2], 1, 0, 0, 0], qposAddress);
}

function cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function rotationWxyz(q) {
  const [w, x, y, z] = q;
  return [
    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
  ];
}

// Port of sim2sim_mujoco_teacher.analytic_bounce (impulse model fitted to Isaac bounces).
function analyticBounce(v, w, radius, params) {
  const vn = v[2];
  const vt = [v[0], v[1], 0];
  const lever = [0, 0, -radius];
  const spin = cross(w, lever);
  const slip = [vt[0] + spin[0], vt[1] + spin[1], 0];
  const speed = Math.hypot(slip[0], slip[1]);
  let dvt = [0, 0, 0];
  if (speed > 1e-9) {
    const jn = (1 + params.restitution) * -vn;
    const magnitude = Math.min(params.friction * jn, (1 + params.beta) * 2 / 7 * speed);
    dvt = [-magnitude * slip[0] / speed, -magnitude * slip[1] / speed, 0];
  }
  const dw = cross(lever, dvt);
  const k = 5 / (2 * radius * radius);
  return {
    v: [params.tangent_scale * (vt[0] + dvt[0]), params.tangent_scale * (vt[1] + dvt[1]), -params.restitution * vn],
    w: [w[0] + k * dw[0], w[1] + k * dw[1], w[2] + k * dw[2]],
  };
}

function applyAnalyticGround() {
  const params = manifest.ball_ground;
  if (params?.model !== "analytic") return false;
  const radius = manifest.balls.radius_m;
  let contacted = false;
  for (const [ballIndex, ball] of manifest.balls.slots.entries()) {
    const qa = ball.qpos_adr;
    const va = ball.qvel_adr;
    const z = data.qpos[qa + 2];
    if (z >= radius + 1e-3) continue;
    if (ballLaunched && ballIndex === 0) contacted = true;
    if (data.qvel[va + 2] < -params.min_impact_speed_m_s) {
      const rot = rotationWxyz(Array.from(data.qpos.subarray(qa + 3, qa + 7)));
      const local = Array.from(data.qvel.subarray(va + 3, va + 6));
      const worldW = rot.map((row) => row[0] * local[0] + row[1] * local[1] + row[2] * local[2]);
      const result = analyticBounce(Array.from(data.qvel.subarray(va, va + 3)), worldW, radius, params);
      const bodyW = [0, 1, 2].map((col) => rot[0][col] * result.w[0] + rot[1][col] * result.w[1] + rot[2][col] * result.w[2]);
      data.qpos[qa + 2] = Math.max(z, radius);
      data.qvel.set([...result.v, ...bodyW], va);
    } else {
      data.qpos[qa + 2] = Math.max(z, radius);
      data.qvel[va + 2] = Math.max(data.qvel[va + 2], 0);
    }
  }
  return contacted;
}

function resetControllerDelay() {
  const action = manifest.action;
  delayedTargets = Array.from({ length: action.delay_physics_steps }, () => Float64Array.from(action.default_joint_pos));
}

function emptyContactSummary() {
  return { leftSubsteps: 0, rightSubsteps: 0, bodySubsteps: 0, groundSubsteps: 0,
    handHandSubsteps: 0, leftForceMaxN: 0, rightForceMaxN: 0, bodyForceMaxN: 0, groundForceMaxN: 0,
    nconMax: 0, ballContactCount: 0, ballContactPairs: [] };
}

function palmPositions() {
  const offsets = [[0, -0.0087001175, -0.132], [0, 0.0087001175, -0.132]];
  return wristBodyIds.map((bodyId, index) => {
    const origin = data.xpos.subarray(bodyId * 3, bodyId * 3 + 3);
    const rotation = data.xmat.subarray(bodyId * 9, bodyId * 9 + 9);
    const offset = offsets[index];
    return [0, 1, 2].map((row) => origin[row]
      + rotation[row * 3] * offset[0]
      + rotation[row * 3 + 1] * offset[1]
      + rotation[row * 3 + 2] * offset[2]);
  });
}

function updateCatchMetrics(contacts) {
  const ball = manifest.balls.slots[0];
  const palms = palmPositions();
  const ballPosition = Array.from(data.qpos.subarray(ball.qpos_adr, ball.qpos_adr + 3));
  const ballVelocity = Array.from(data.qvel.subarray(ball.qvel_adr, ball.qvel_adr + 3));
  const dt = previousPalmTime === null ? 0 : data.time - previousPalmTime;
  const palmVelocity = previousPalmPositions && dt > 0
    ? [0, 1, 2].map((axis) => palms.reduce((sum, palm, index) =>
      sum + (palm[axis] - previousPalmPositions[index][axis]) / dt / 2, 0))
    : [0, 0, 0];
  const relativeSpeed = Math.hypot(...ballVelocity.map((value, axis) => value - palmVelocity[axis]));
  const midpoint = [0, 1, 2].map((axis) => (palms[0][axis] + palms[1][axis]) / 2);
  const midpointDistance = Math.hypot(...ballPosition.map((value, axis) => value - midpoint[axis]));
  const maxPalmDistance = Math.max(...palms.map((palm) => Math.hypot(
    ...ballPosition.map((value, axis) => value - palm[axis]),
  )));
  const bodyContact = contacts.bodySubsteps > 0;
  const groundContact = contacts.groundSubsteps > 0;
  const handHandContact = contacts.handHandSubsteps > 0;
  if (ballLaunched && bodyContact && catchMetrics.bestHoldS < 1.2) catchMetrics.bodyAssisted = true;
  const clean = ballLaunched
    && contacts.leftSubsteps === 8
    && contacts.rightSubsteps === 8
    && !bodyContact && !groundContact && !handHandContact && !catchMetrics.bodyAssisted
    && relativeSpeed <= 0.15
    && midpointDistance <= 0.20
    && maxPalmDistance <= 0.24;
  catchMetrics.currentHoldS = clean ? catchMetrics.currentHoldS + manifest.physics.policy_dt_s : 0;
  catchMetrics.bestHoldS = Math.max(catchMetrics.bestHoldS, catchMetrics.currentHoldS);
  catchMetrics.caught1p2s = catchMetrics.bestHoldS >= 1.2 - 1e-6;
  catchMetrics.clean = clean;
  catchMetrics.relativeSpeedMps = relativeSpeed;
  catchMetrics.midpointDistanceM = midpointDistance;
  catchMetrics.maxPalmDistanceM = maxPalmDistance;
  catchMetrics.ballPosition = ballPosition;
  catchMetrics.palmPositions = palms;
  previousPalmPositions = palms;
  previousPalmTime = data.time;
}

function sampleContacts() {
  const summary = emptyContactSummary();
  summary.nconMax = data.ncon;
  const contactVector = data.contact;
  const wristSet = new Set(wristBodyIds);
  const handHand = new Set();
  let left = false;
  let right = false;
  let body = false;
  let ground = false;
  let leftForce = 0;
  let rightForce = 0;
  let bodyForce = 0;
  let groundForce = 0;

  try {
    for (let contactIndex = 0; contactIndex < contactVector.size(); contactIndex++) {
      const contact = contactVector.get(contactIndex);
      if (!contact) continue;
      try {
        const geom1 = contact.geom1;
        const geom2 = contact.geom2;
        const body1 = Number(model.geom_bodyid[geom1]);
        const body2 = Number(model.geom_bodyid[geom2]);
        const force = contactForceBuffer.GetView();
        force.fill(0);
        mujoco.mj_contactForce(model, data, contactIndex, contactForceBuffer);
        const currentForce = contactForceBuffer.GetView();
        const magnitude = Math.hypot(currentForce[0], currentForce[1], currentForce[2]);
        const activeBallGeomId = ballGeomIds[0];
        const hasBall = geom1 === activeBallGeomId || geom2 === activeBallGeomId;

        if (hasBall) {
          summary.ballContactCount++;
          summary.ballContactPairs.push([geom1, geom2]);
          const otherGeom = geom1 === activeBallGeomId ? geom2 : geom1;
          const otherBody = geom1 === activeBallGeomId ? body2 : body1;
          if (otherBody === wristBodyIds[0] && magnitude > 0.1) {
            left = true;
            leftForce = Math.max(leftForce, magnitude);
          } else if (otherBody === wristBodyIds[1] && magnitude > 0.1) {
            right = true;
            rightForce = Math.max(rightForce, magnitude);
          } else if (groundGeomIds.has(otherGeom) && magnitude > 0.1) {
            ground = true;
            groundForce = Math.max(groundForce, magnitude);
          } else if (robotBodyIds.has(otherBody) && magnitude > 0.5) {
            body = true;
            bodyForce = Math.max(bodyForce, magnitude);
          }
        } else if (magnitude > 0.1 && wristSet.has(body1) && wristSet.has(body2) && body1 !== body2) {
          handHand.add(`${Math.min(body1, body2)}:${Math.max(body1, body2)}`);
        }
      } finally {
        contact.delete();
      }
    }
  } finally {
    contactVector.delete();
  }

  if (left) summary.leftSubsteps = 1;
  if (right) summary.rightSubsteps = 1;
  if (body) summary.bodySubsteps = 1;
  if (ground) summary.groundSubsteps = 1;
  if (handHand.size) summary.handHandSubsteps = 1;
  summary.leftForceMaxN = leftForce;
  summary.rightForceMaxN = rightForce;
  summary.bodyForceMaxN = bodyForce;
  summary.groundForceMaxN = groundForce;
  return summary;
}

function resetScene() {
  mujoco.mj_resetData(model, data);
  data.qpos.set(manifest.initial_state.qpos);
  data.qvel.set(manifest.initial_state.qvel);
  resetControllerDelay();
  mujoco.mj_forward(model, data);
  ballLaunched = false;
  lastContactSummary = emptyContactSummary();
  catchMetrics = { currentHoldS: 0, bestHoldS: 0, caught1p2s: false, bodyAssisted: false,
    clean: false, relativeSpeedMps: 0, midpointDistanceM: 0, maxPalmDistanceM: 0,
    ballPosition: [], palmPositions: [] };
  previousPalmPositions = palmPositions();
  previousPalmTime = data.time;
}

function launchBall(message) {
  const ball = manifest.balls.slots[message.ballId ?? 0];
  if (!ball) throw new Error("Unknown ball slot");
  const preset = message.presetOverride ?? manifest.launch_presets?.[message.presetIndex];
  if (!preset) throw new Error("Unknown launch preset");
  if (preset.clip !== manifest.launch_presets?.[message.presetIndex]?.clip) {
    throw new Error("Launch preset override does not match the selected clip");
  }
  const root = manifest.root.qpos_adr;
  const [w, x, y, z] = data.qpos.subarray(root + 3, root + 7);
  const yaw = Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
  const c = Math.cos(yaw);
  const s = Math.sin(yaw);
  const rotate = (v) => [c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]];
  const offset = rotate(preset.position_yaw_frame_m);
  identityBallPose(ball.qpos_adr, [data.qpos[root] + offset[0], data.qpos[root + 1] + offset[1], offset[2]]);
  data.qvel.set([...rotate(preset.velocity_yaw_frame_m_s), ...rotate(preset.angular_velocity_yaw_frame_rad_s)], ball.qvel_adr);
  ballLaunched = true;
  catchMetrics.currentHoldS = 0;
  catchMetrics.bestHoldS = 0;
  catchMetrics.caught1p2s = false;
  catchMetrics.bodyAssisted = false;
  lastContactSummary = emptyContactSummary();
  mujoco.mj_forward(model, data);
  previousPalmPositions = palmPositions();
  previousPalmTime = data.time;
}

function applyPolicyAction(actionValues) {
  if (!actionValues || actionValues.length !== manifest.action.joint_names.length) {
    throw new Error(`Expected ${manifest.action.joint_names.length} joint actions`);
  }
  const action = manifest.action;
  const unbounded = manifest.joint_target_limits === "unbounded";
  const commandTarget = new Float64Array(action.joint_names.length);
  const contactTotals = emptyContactSummary();
  for (let joint = 0; joint < commandTarget.length; joint++) {
    const raw = Number(actionValues[joint]);
    const target = action.default_joint_pos[joint] + action.action_scale[joint] * raw;
    commandTarget[joint] = unbounded ? target : clamp(
      target,
      action.joint_limits_rad[joint][0],
      action.joint_limits_rad[joint][1],
    );
  }

  for (let step = 0; step < 8; step++) {
    let target = commandTarget;
    if (action.delay_physics_steps) {
      delayedTargets.push(Float64Array.from(commandTarget));
      target = delayedTargets.shift();
    }
    for (let joint = 0; joint < target.length; joint++) {
      const q = data.qpos[action.qpos_addresses[joint]];
      const qd = data.qvel[action.qvel_addresses[joint]];
      data.ctrl[joint] = clamp(
        action.kp[joint] * (target[joint] - q) - action.kd[joint] * qd,
        -action.effort_limit_nm[joint],
        action.effort_limit_nm[joint],
      );
    }
    mujoco.mj_step(model, data);
    const analyticGroundContact = applyAnalyticGround();
    const substepContacts = sampleContacts();
    if (analyticGroundContact) substepContacts.groundSubsteps = 1;
    for (const key of Object.keys(contactTotals)) {
      if (key.endsWith("Substeps")) contactTotals[key] += substepContacts[key];
      else if (key === "ballContactCount") contactTotals[key] += substepContacts[key];
      else if (key === "ballContactPairs") {
        contactTotals[key] = [...contactTotals[key], ...substepContacts[key]].slice(-16);
      }
      else contactTotals[key] = Math.max(contactTotals[key], substepContacts[key]);
    }
  }
  lastContactSummary = contactTotals;
  updateCatchMetrics(contactTotals);
}

function snapshot() {
  mujoco.mj_kinematics(model, data);
  const root = manifest.root.qpos_adr;
  const ball = manifest.balls.slots[0].qpos_adr;
  const cameraBodyOffset = cameraBodyId * 3;
  const cameraQuaternionOffset = cameraBodyId * 4;
  const poses = new Float32Array(model.ngeom * 12);
  for (let geom = 0; geom < model.ngeom; geom++) {
    const offset = geom * 12;
    poses.set(data.geom_xpos.subarray(geom * 3, geom * 3 + 3), offset);
    poses.set(data.geom_xmat.subarray(geom * 9, geom * 9 + 9), offset + 3);
  }
  return {
    time: data.time,
    root: Array.from(data.qpos.subarray(root, root + 3)),
    rootQuaternion: Array.from(data.qpos.subarray(root + 3, root + 7)),
    ball: Array.from(data.qpos.subarray(ball, ball + 3)),
    camera: {
      position: Array.from(data.xpos.subarray(cameraBodyOffset, cameraBodyOffset + 3)),
      quaternionWxyz: Array.from(data.xquat.subarray(cameraQuaternionOffset, cameraQuaternionOffset + 4)),
    },
    qpos: Array.from(data.qpos),
    qvel: Array.from(data.qvel),
    contacts: { ...(lastContactSummary ?? emptyContactSummary()), ballGeomId: ballGeomIds[0] },
    catch: catchMetrics,
    poses,
  };
}

function renderConfig() {
  const meshes = new Map();
  const geoms = [];
  for (let geom = 0; geom < model.ngeom; geom++) {
    const type = model.geom_type[geom];
    const meshId = type === 7 ? model.geom_dataid[geom] : -1;
    if (meshId >= 0 && !meshes.has(meshId)) {
      const vertexStart = model.mesh_vertadr[meshId] * 3;
      const vertexLength = model.mesh_vertnum[meshId] * 3;
      const faceStart = model.mesh_faceadr[meshId] * 3;
      const faceLength = model.mesh_facenum[meshId] * 3;
      meshes.set(meshId, {
        id: meshId,
        vertices: Float32Array.from(model.mesh_vert.subarray(vertexStart, vertexStart + vertexLength)),
        faces: Uint32Array.from(model.mesh_face.subarray(faceStart, faceStart + faceLength)),
      });
    }
    geoms.push({
      id: geom,
      name: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM.value, geom) || `geom_${geom}`,
      bodyId: Number(model.geom_bodyid[geom]),
      contype: Number(model.geom_contype[geom]),
      conaffinity: Number(model.geom_conaffinity[geom]),
      type,
      meshId,
      size: Array.from(model.geom_size.subarray(geom * 3, geom * 3 + 3)),
      color: Array.from(model.geom_rgba.subarray(geom * 4, geom * 4 + 4)),
      ball: geom === ballGeomIds[0],
    });
  }
  return { geoms, meshes: [...meshes.values()] };
}

async function initialize(initialStateOverride = null) {
  const manifestResponse = await fetch(new URL("manifest.json?v=sim2sim3", BUNDLE_ROOT));
  if (!manifestResponse.ok) throw new Error(`Scene manifest HTTP ${manifestResponse.status}`);
  manifest = await manifestResponse.json();
  mujoco = await loadMujoco();
  contactForceBuffer = new mujoco.DoubleBuffer(6);
  const runtimeVersion = mujoco.mj_versionString();
  if (runtimeVersion !== manifest.simulator.replace("MuJoCo ", "")) {
    throw new Error(`MuJoCo version mismatch: bundle ${manifest.simulator}, browser ${runtimeVersion}`);
  }

  // GitHub Pages rejects files over 100 MB, so the public bundle may ship the gzip in parts.
  const modelFiles = manifest.model.parts ?? [manifest.model.file];
  const compressedParts = await Promise.all(modelFiles.map(async (file) => {
    const response = await fetch(new URL(`${file}?v=${manifest.model.sha256.slice(0, 12)}`, BUNDLE_ROOT));
    if (!response.ok) throw new Error(`MJB HTTP ${response.status}`);
    return response.arrayBuffer();
  }));
  const compressed = new Blob(compressedParts);
  if (compressed.size !== manifest.model.bytes) throw new Error("MJB size mismatch");
  const raw = await new Response(compressed.stream().pipeThrough(new DecompressionStream("gzip")))
    .arrayBuffer();
  mujoco.FS.mkdir("/l04");
  mujoco.FS.writeFile(MODEL_PATH, new Uint8Array(raw));
  vfs = new mujoco.MjVFS();
  model = mujoco.MjModel.mj_loadModel(MODEL_PATH, vfs);
  if (!model) throw new Error("MuJoCo could not load the packaged model");
  if (initialStateOverride) {
    const { rootState, jointNames, jointPos, jointVel } = initialStateOverride;
    if (rootState?.length !== 13 || !Array.isArray(jointNames)
      || jointPos?.length !== jointNames.length || jointVel?.length !== jointNames.length) {
      throw new Error("Diagnostic root or joint state dimensions do not match the scene");
    }
    const initialQpos = Array.from(manifest.initial_state.qpos);
    const initialQvel = Array.from(manifest.initial_state.qvel);
    const rootQpos = manifest.root.qpos_adr;
    const rootQvel = manifest.root.qvel_adr;
    initialQpos.splice(rootQpos, 7, ...rootState.slice(0, 7));
    const rootRotation = rotationWxyz(rootState.slice(3, 7));
    const worldAngular = rootState.slice(10, 13);
    const bodyId = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, "base_link");
    const comOffset = model.body_ipos.subarray(bodyId * 3, bodyId * 3 + 3);
    const offsetWorld = rootRotation.map((row) => row.reduce((sum, value, axis) => sum + value * comOffset[axis], 0));
    const rootLinear = rootState.slice(7, 10).map((value, axis) => value - cross(worldAngular, offsetWorld)[axis]);
    const rootAngular = [0, 1, 2].map((axis) => rootRotation.reduce((sum, row, worldAxis) =>
      sum + row[axis] * worldAngular[worldAxis], 0));
    initialQvel.splice(rootQvel, 6, ...rootLinear, ...rootAngular);
    let matchedJoints = 0;
    for (let index = 0; index < jointNames.length; index++) {
      const jointId = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT.value, jointNames[index]);
      if (jointId < 0) continue;
      initialQpos[model.jnt_qposadr[jointId]] = Number(jointPos[index]);
      initialQvel[model.jnt_dofadr[jointId]] = Number(jointVel[index]);
      matchedJoints++;
    }
    if (matchedJoints < manifest.action.joint_names.length) {
      throw new Error(`Diagnostic state matched only ${matchedJoints} scene joints`);
    }
    manifest.initial_state.qpos = initialQpos;
    manifest.initial_state.qvel = initialQvel;
    manifest.diagnostic_initial_state_matched_joints = matchedJoints;
  }
  cameraBodyId = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, manifest.depth_camera.parent_link);
  if (cameraBodyId < 0) throw new Error(`Camera parent body not found: ${manifest.depth_camera.parent_link}`);
  wristBodyIds = ["left_wrist_yaw_link", "right_wrist_yaw_link"].map((name) =>
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, name));
  if (wristBodyIds.some((id) => id < 0)) throw new Error("MuJoCo hand bodies were not found");
  ballGeomIds = manifest.balls.slots.map(({ id }) =>
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM.value, `ball_${id}_geom`));
  if (ballGeomIds.some((id) => id < 0)) throw new Error("MuJoCo ball collision geometries were not found");
  const robotRootId = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, "base_link");
  robotBodyIds = new Set();
  groundGeomIds = new Set();
  for (let bodyId = 0; bodyId < model.nbody; bodyId++) {
    let ancestor = bodyId;
    while (ancestor > 0 && ancestor !== robotRootId) ancestor = Number(model.body_parentid[ancestor]);
    if (ancestor === robotRootId && !wristBodyIds.includes(bodyId)) robotBodyIds.add(bodyId);
  }
  for (let geomId = 0; geomId < model.ngeom; geomId++) {
    const name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM.value, geomId) || "";
    if (/^(sim_ground|ground|floor)$/i.test(name)) groundGeomIds.add(geomId);
  }
  data = new mujoco.MjData(model);
  resetScene();
  const render = renderConfig();
  const renderBuffers = render.meshes.flatMap((mesh) => [mesh.vertices.buffer, mesh.faces.buffer]);
  const state = snapshot();
  self.postMessage({ type: "ready", version: runtimeVersion, shape: manifest.model_shape,
    cameraBodyId, render, state,
    diagnosticInitialStateMatchedJoints: manifest.diagnostic_initial_state_matched_joints },
    renderBuffers);
}

async function handle(message) {
  if (message.type === "init") return initialize(message.initialState);
  if (!model || !data) throw new Error("MuJoCo scene is not initialized");
  if (message.type === "reset") {
    resetScene();
    self.postMessage({ type: "state", state: snapshot() });
  } else if (message.type === "replay_state") {
    if (message.qpos?.length !== model.nq || message.qvel?.length !== model.nv
      || ![...message.qpos, ...message.qvel].every(Number.isFinite)) {
      throw new Error("Replay frame dimensions or values are invalid");
    }
    data.qpos.set(message.qpos);
    data.qvel.set(message.qvel);
    if (Number.isFinite(message.time)) data.time = message.time;
    mujoco.mj_forward(model, data);
    const state = snapshot();
    if (message.catch) state.catch = message.catch;
    if (message.contacts) state.contacts = message.contacts;
    self.postMessage({ type: "state", state }, [state.poses.buffer]);
  } else if (message.type === "step") {
    if (message.launch) launchBall(message.launch);
    applyPolicyAction(message.action);
    const state = snapshot();
    self.postMessage({ type: "state", state }, [state.poses.buffer]);
  } else {
    throw new Error(`Unknown MuJoCo command: ${message.type}`);
  }
  if (message.type === "reset") {
    const state = snapshot();
    self.postMessage({ type: "state", state }, [state.poses.buffer]);
  }
}

self.onmessage = ({ data: message }) => {
  commandQueue = commandQueue.then(() => handle(message)).catch((error) => {
    self.postMessage({ type: "error", message: error.message || String(error) });
  });
};