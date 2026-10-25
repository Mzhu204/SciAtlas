// One bundle, one three.js instance, zero CDN calls at demo time.
// Rebuild:  cd vendor && npm install && npm run build
import * as THREE from 'three';
import ForceGraph3D from '3d-force-graph';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';

window.ATLAS_VENDOR = { THREE, ForceGraph3D, UnrealBloomPass };
