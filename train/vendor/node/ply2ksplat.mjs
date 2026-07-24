// Convert scene.ply -> scene.ksplat using the same library the viewer loads at
// runtime, so the compressed .ksplat format always matches exactly what
// gaussian-splats-3d's own KSplatLoader expects (no hand-rolled binary format).
import fs from 'fs';
import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d';

const [, , inPath, outPath, shDegreeArg] = process.argv;
if (!inPath || !outPath) {
  console.error('usage: node ply2ksplat.mjs <in.ply> <out.ksplat> [shDegree=3]');
  process.exit(1);
}
const shDegree = shDegreeArg ? parseInt(shDegreeArg, 10) : 3;

const plyBuffer = fs.readFileSync(inPath).buffer;
const splatArray = GaussianSplats3D.PlyParser.parseToUncompressedSplatArray(plyBuffer, shDegree);
console.log(`parsed ${splatArray.splatCount.toLocaleString()} splats from ${inPath}`);

const generator = GaussianSplats3D.SplatBufferGenerator.getStandardGenerator(1, 1);
const splatBuffer = generator.generateFromUncompressedSplatArray(splatArray);

fs.writeFileSync(outPath, Buffer.from(splatBuffer.bufferData));
console.log(`wrote ${outPath} (${(splatBuffer.bufferData.byteLength / 1e6).toFixed(1)} MB)`);
