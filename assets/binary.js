// Define the main function for initializing and running WebGPU
async function main() {
    const adapter = await navigator.gpu?.requestAdapter();
    const device = await adapter?.requestDevice();
    if (!device) {
      fail('WebGPU is not supported in this browser.');
      return;
    }
  
    // Get a WebGPU context from the canvas and configure it
    const canvas = document.querySelector('canvas');
    const context = canvas.getContext('webgpu');
    const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
      device,
      format: presentationFormat,
    });
  
    const computeModule = device.createShaderModule({
        label: 'histogram shader',
        code: `
          @group(0) @binding(0) var ourTexture: texture_2d<f32>;
          @group(0) @binding(1) var<storage, read_write> pixelCountBuffer: array<atomic<u32>>;
    
          @compute @workgroup_size(8, 8) fn cs(@builtin(global_invocation_id) id: vec3u) {
              let size = textureDimensions(ourTexture);
              if (id.x >= size.x || id.y >= size.y) {
                  return;
              }
              let floatId: vec3f = vec3f(id); // id is vec3u
              let scale: f32 = 0.5;          
              let scaledId: vec3f = floatId * scale; // Perform the operation in float space
    
              let color = textureLoad(ourTexture, vec2u(id.xy), 0);
              if (color.r > 0.0 || color.g > 0.0 || color.b > 0.0) {
                  atomicAdd(&pixelCountBuffer[0], 1);
              }
          }
        `,
      });
    
    
      const computePipeline = device.createComputePipeline({
          layout: 'auto',
          compute: {
              module: computeModule,
              entryPoint: 'cs',
          },
      });

        
    canvas.width = Math.max(1, Math.min(canvas.clientWidth, device.limits.maxTextureDimension2D));
    canvas.height = Math.max(1, Math.min(canvas.clientHeight, device.limits.maxTextureDimension2D));

    const texture = device.createTexture({
        size: [canvas.width, canvas.height, 1],
        format: presentationFormat,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });


    const textureView = texture.createView();

    const accumulationTexture = device.createTexture({
        size: [canvas.width, canvas.height, 1],
        format: presentationFormat,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });

    const accumulationView = accumulationTexture.createView();

    
  const module = device.createShaderModule({
    code: `
      struct Star {
        x: f32,
        y: f32,
        z: f32,
        n: f32
      };

      struct InOut {
        @builtin(position) position: vec4f,
        @location(0) tex: vec2f,
      };

      struct TransformData {
          mvp: mat4x4f,
      };

      struct TimeData {
        time: f32,
        npix: f32,
      };


      @group(0) @binding(0) var<storage, read> stars: array<Star>;
      @group(0) @binding(1) var<uniform> timeData: TimeData;
      // @group(0) @binding(1) var<uniform> transformData: TransformData;


      @vertex fn vs1( @builtin(vertex_index) vertex : u32 ) -> InOut {

        // Quad
        let pos = array(
          vec2f( -1, -1),
          vec2f( -1,  1),
          vec2f(  1, -1),
          vec2f(  1,  1),
          vec2f(  1, -1),
          vec2f( -1,  1)
        );

        var star = stars[vertex/6];
        star.x += star.n * sin(timeData.time)*0.6+35;
        star.z +=  star.n * sin(timeData.time)*0.1;


        let size = 0.3;
        
        let x = size*pos[vertex%6].x + star.x; 
        let y = size*pos[vertex%6].y + star.y; 
        let z = star.z;

        var inOut: InOut;
        inOut.position = vec4f( x, y, z, 1.0);
        inOut.tex = pos[vertex%6];
        return inOut;


      };

      @vertex fn vs2(@builtin(vertex_index) vertexIndex: u32) -> InOut {
    // Quad vertex positions
    let pos = array(
        vec2f(-1, -1),
        vec2f(-1,  1),
        vec2f( 1, -1),
        vec2f( 1,  1),
        vec2f( 1, -1),
        vec2f(-1,  1)
    );

    var star = stars[vertexIndex / 6];

    let size = 0.02;
    let x = size * pos[vertexIndex % 6].x - 0.8; // Increment x with time

    let y = size * pos[vertexIndex % 6].y + 0.00001 * timeData.npix - 0.9; 
    let z = star.z;



        var inOut: InOut;
        inOut.position = vec4f( x, y, z, 1.0);
        inOut.tex = pos[vertexIndex%6];
        return inOut;
      };


      @fragment fn fs(inOut: InOut) -> @location(0) vec4f {
        // let phi = atan2(inOut.tex.y,inOut.tex.x);
        var r = 1.0-length(inOut.tex);
        let a = pow(r,4.0);
        let alpha = 2*inOut.position.z;
        let f = pow(1.1*r,8.0) + r*a;
        return vec4f(1,1-alpha,1-alpha,f);
      }

    `,
  });

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' }},
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' }},
    ],
});

const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
});

const pipeline1 = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: { module, entryPoint: 'vs1' },
    fragment: {
        module,
        entryPoint: 'fs',
        targets: [{
            format: presentationFormat,
            blend: {
                color: {
                    srcFactor: 'src-alpha',
                    dstFactor: 'one-minus-src-alpha',
                    operation: 'add',
                },
                alpha: {
                    srcFactor: 'one',
                    dstFactor: 'zero',
                    operation: 'add',
                },
            },
        }],
    },
});


const pipeline2 = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: { module, entryPoint: 'vs2' },
    fragment: {
        module,
        entryPoint: 'fs',
        targets: [{
            format: presentationFormat,
            blend: {
                color: {
                    srcFactor: 'src-alpha',
                    dstFactor: 'one-minus-src-alpha',
                    operation: 'add',
                },
                alpha: {
                    srcFactor: 'one',
                    dstFactor: 'zero',
                    operation: 'add',
                },
            },
        }],
    },
});


const pixelCountBuffer = device.createBuffer({
    size: 4, // 4 bytes for one atomic<u32>
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
});


const readbackBuffer = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
});


  
const timeBufferSize = 4 + 4;


const timeBuffer = device.createBuffer({
  size: timeBufferSize,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});
  
    // const timeValues = new Float32Array(timeBufferSize / 4);
  
    const staticStorageBufferSize = 4 * 4 * 100;
  
    const staticStorageBuffer = device.createBuffer({
      label: 'static storage for objects',
      size: staticStorageBufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    
     const staticStorageValues = new Float32Array([
      [-35.05,0.01,0.1,1],
      [-35.05,0.01,0.1,-1],
      ].flat());
  
    device.queue.writeBuffer(staticStorageBuffer, 0, staticStorageValues);

    const computeBindGroup = device.createBindGroup({
        layout: computePipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: textureView },
            { binding: 1, resource: { buffer: pixelCountBuffer } },
        ],
    });
  
  
  
    const bindGroup = device.createBindGroup({
      layout: bindGroupLayout, 
      entries: [
          { binding: 0, resource: { buffer: staticStorageBuffer }},
          { binding: 1, resource: { buffer: timeBuffer }}  ],
  });
  
  
    const renderPassDescriptor = {
      label: 'our basic canvas renderPass',
      colorAttachments: [
        {
          // view: <- to be filled out when we render
          clearValue: [0, 0, 0, 1],
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    };

    const renderPassDescriptorPipeline2 = {
        label: 'pipeline2 render pass',
        colorAttachments: [
            {
                view: context.getCurrentTexture().createView(), // Render directly to canvas
                loadOp: 'load', // Keep the previous frame's dots
                storeOp: 'store', // Save the updated frame
            },
        ],
    };
    
  
let npix = 0.0;
// let isBufferMapped = false;

let isBufferMapped = false;

async function processPixelCount() {
    if (isBufferMapped) {
        // console.warn("Skipping processPixelCount as buffer is already mapped.");
        return;
    }

    isBufferMapped = true;

    // Wait for GPU work to finish before mapping
    await device.queue.onSubmittedWorkDone();

    try {
        await readbackBuffer.mapAsync(GPUMapMode.READ);
        const arrayBuffer = readbackBuffer.getMappedRange();
        const pixelCount = new Uint32Array(arrayBuffer)[0];
        console.log('Pixel Count:', pixelCount);

        npix = pixelCount; // Update the npix value
    } catch (err) {
        console.error("Error mapping buffer:", err);
    } finally {
        readbackBuffer.unmap();
        isBufferMapped = false;
    }
}
    var width = 100.0;
    var height = 100.0;
    var time = 0.0;
  // var npix = 0.0;
  function render() {
    if (isBufferMapped) {
        requestAnimationFrame(render); // Skip frame if buffer is mapped
        return;
    }

    // Reset the pixel count buffer
    const zeroBuffer = new Uint32Array([0]);
    device.queue.writeBuffer(pixelCountBuffer, 0, zeroBuffer);

    // Update time for animation
    time += 0.01;
    const timeDataArray = new Float32Array([time, npix]);
    device.queue.writeBuffer(timeBuffer, 0, timeDataArray);

    const encoder = device.createCommandEncoder();

    // Step 1: make texture
    renderPassDescriptor.colorAttachments[0].view = texture.createView(); // Render to custom texture
    const computeRenderPass = encoder.beginRenderPass(renderPassDescriptor);
    computeRenderPass.setPipeline(pipeline1);
    computeRenderPass.setBindGroup(0, bindGroup);
    computeRenderPass.draw(6 * 100); // Adjust for your specific geometry
    computeRenderPass.end();

    renderPassDescriptorPipeline2.colorAttachments[0].view = context.getCurrentTexture().createView();
    const renderPass = encoder.beginRenderPass(renderPassDescriptorPipeline2);
    // computeRenderPass.setPipeline(pipeline2);
    // computeRenderPass.setBindGroup(0, bindGroup);
    // computeRenderPass.draw(6 * 100); // Adjust for your specific geometry

    renderPass.setPipeline(pipeline2);
    renderPass.setBindGroup(0, bindGroup);
    renderPass.draw(6 * 100); // Adjust for your specific geometry
    renderPass.end();


    // computeRenderPass.end();


    device.queue.submit([encoder.finish()]);

    // Step 2: Render to display
    renderPassDescriptor.colorAttachments[0].view = context.getCurrentTexture().createView(); 
    const displayEncoder = device.createCommandEncoder();
    const displayRenderPass = displayEncoder.beginRenderPass(renderPassDescriptor);
    displayRenderPass.setPipeline(pipeline1);
    displayRenderPass.setBindGroup(0, bindGroup);
    displayRenderPass.draw(6 * 100); 
    displayRenderPass.setPipeline(pipeline2);
    displayRenderPass.setBindGroup(0, bindGroup);
    displayRenderPass.draw(6 * 100); 
    displayRenderPass.end();


    device.queue.submit([displayEncoder.finish()]);

    // Step 3: Compute Pass
    const computeEncoder = device.createCommandEncoder();
    const computePass = computeEncoder.beginComputePass();
    computePass.setPipeline(computePipeline);
    computePass.setBindGroup(0, computeBindGroup);
    const workgroupSize = 8; // Matches @workgroup_size(8, 8) in the shader
    computePass.dispatchWorkgroups(
        Math.ceil(canvas.width / workgroupSize),
        Math.ceil(canvas.height / workgroupSize)
    );
    computePass.end();

    computeEncoder.copyBufferToBuffer(pixelCountBuffer, 0, readbackBuffer, 0, 4);
    device.queue.submit([computeEncoder.finish()]);

    // Process compute results
    device.queue.onSubmittedWorkDone().then(processPixelCount);
    // device.queue.writeBuffer(transformBuffer, 0, new Float32Array([npix]));


    // Request next frame
    requestAnimationFrame(render);
}
requestAnimationFrame(render);



    const observer = new ResizeObserver(entries => {
      for (const entry of entries) {
        const canvas = entry.target;
        const width = entry.contentBoxSize[0].inlineSize;
        const height = entry.contentBoxSize[0].blockSize;
        canvas.width = Math.max(1, Math.min(width, device.limits.maxTextureDimension2D));
        canvas.height = Math.max(1, Math.min(height, device.limits.maxTextureDimension2D));
        // re-render
        render();
      }
    });
    observer.observe(canvas);
  }
  
  function fail(msg) {
    alert(msg);
  }
  
  main();
      