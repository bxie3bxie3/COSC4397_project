"use strict";

async function LoadShader(device) 
{
    const response = await fetch("shader_matmult_opt.wgsl");
    const shader_code = await response.text();
	const shader_module = device.createShaderModule({
		code: shader_code,
	});
    return shader_module;
}

function CreateRandSquareMatrix(device, matrix_size)
{
	let num_timestamps = 2;
	const script_matrix = new Float32Array(num_timestamps + matrix_size * matrix_size);
	for (let i = num_timestamps; i < script_matrix.length; i++) {
		script_matrix[i] = Math.random();
	}
	script_matrix[0] = matrix_size;
	script_matrix[1] = matrix_size;
	const gpu_buf = device.createBuffer({
		mappedAtCreation: true,
		size: script_matrix.byteLength,
		usage: GPUBufferUsage.STORAGE,
	});
	const array_buf = gpu_buf.getMappedRange();
	new Float32Array(array_buf).set(script_matrix);
	gpu_buf.unmap();
	return [script_matrix, gpu_buf];
}

function CreateResultMatrix(device, matrix_size)
{
	let num_timestamps = 2;
	const buf_size = Float32Array.BYTES_PER_ELEMENT * (num_timestamps + matrix_size * matrix_size);
	const gpu_buf = device.createBuffer({
		size: buf_size,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
	return [buf_size, gpu_buf];
}
		
function CreatePipeline(device, shader_module)
{
	const bindGroupLayout = device.createBindGroupLayout({
		entries: [
			{
				binding: 0,
				visibility: GPUShaderStage.COMPUTE,
				buffer: {
					type: "read-only-storage",
				},
			},
			{
				binding: 1,
				visibility: GPUShaderStage.COMPUTE,
				buffer: {
					type: "read-only-storage",
				},
			},
			{
				binding: 2,
				visibility: GPUShaderStage.COMPUTE,
				buffer: {
					type: "storage",
				},
			},
		],
	});
	
	const computePipeline = device.createComputePipeline({
		layout: device.createPipelineLayout({
			bindGroupLayouts: [bindGroupLayout],
		}),
		compute: {
			module: shader_module,
			entryPoint: "main",
		},
	});
	return computePipeline;
}

function GPUCopyResultsToRead(device, commandEncoder, gpu_buf_write, num_bytes)
{
	const gpu_buf_read = device.createBuffer({
		size: num_bytes,
		usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
	});
	commandEncoder.copyBufferToBuffer(gpu_buf_write, 0, gpu_buf_read, 0, num_bytes);
	return gpu_buf_read;
}

async function ReadGPUTimeDelta(gpu_buf_timestamp_read)
{
	await gpu_buf_timestamp_read.mapAsync(GPUMapMode.READ);
	const mapped_range_timestamps = gpu_buf_timestamp_read.getMappedRange();
	const gpu_timestamps = new BigUint64Array(mapped_range_timestamps);
	const gpu_ns = Number(gpu_timestamps[1] - gpu_timestamps[0]);
    gpu_buf_timestamp_read.unmap();
	return gpu_ns;
}

function BindDataBuffers(device, passEncoder, computePipeline, gpu_buf_left_matrix, gpu_buf_right_matrix, gpu_buf_result_matrix_write)
{
	const bindGroup = device.createBindGroup({
		layout: computePipeline.getBindGroupLayout(0),
		entries: [
			{
				binding: 0,
				resource: {
					buffer: gpu_buf_left_matrix,
				},
			},
			{
				binding: 1,
				resource: {
					buffer: gpu_buf_right_matrix,
				},
			},
			{
				binding: 2,
				resource: {
					buffer: gpu_buf_result_matrix_write,
				},
			},
		],
	});
	passEncoder.setBindGroup(0, bindGroup);
}

function SetUpTimestampQuery(device)
{
	let num_timestamps = 2;
	var timestamp_query_set = device.createQuerySet({
		type: "timestamp",
		count: num_timestamps,
	});
	const timestamp_command = {
		querySet: timestamp_query_set,
		beginningOfPassWriteIndex: 0, // Record the timestamp in index 0 at the beginning of the pass
		endOfPassWriteIndex: 1, // Record the timestamp in index 1 when the end of the pass
	};
	return [timestamp_query_set, timestamp_command];
}

function SetUpWorkgroups(passEncoder, left_matrix_rows, right_matrix_cols, BLOCK_SIZE, repetitions)
{		
	const blockX = Math.ceil(left_matrix_rows / BLOCK_SIZE);
	const blockY = Math.ceil(right_matrix_cols / BLOCK_SIZE);
	for (let i = 0; i < repetitions; i++) {
		passEncoder.dispatchWorkgroups(blockX, blockY);
	}
}
		
function DisplayResult(content)
{
	const div_result = document.getElementById("result");
	const pre_result = document.createElement("pre");
	pre_result.textContent = content;
	div_result.prepend(pre_result);
}
		
(async () => {
	
	DisplayResult("Debug: Javascript is running");
	
	// Get the high-performance adapter to the GPU
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
		console.log("WebGPU not supported");
		DisplayResult("WebGPU not supported");
        return;
    }
	DisplayResult("Debug: WebGPU is supported");
	
	// Make sure adapter can time the start and end of the compute pass
    if (!adapter.features.has("timestamp-query")) {
		console.log("Adapter cannot timestamp. Can't use for comparison");
		DisplayResult("Adapter cannot timestamp. Can't use for comparison");
        return;
    }	
	DisplayResult("Debug: Adapter supports timestamp query");
		
	// Get the logical device that provides the actual access to all WebGPU functionality
    const device = await adapter.requestDevice({
        requiredFeatures: ["timestamp-query"],
    });
	
	// Create execution button
    const controlDiv = document.getElementById("execute");
    const runButton = document.createElement("button");
    runButton.innerText = "Run and verify";
    controlDiv.appendChild(runButton);
    controlDiv.appendChild(document.createElement("br"));

    runButton.onclick = async () => {
		let matrix_size = 2048;
        let num_repeats = 20;
        await GPU_MatrixMult(matrix_size, num_repeats);
    };
    const GPU_MatrixMult = async (matrix_size, repetitions) => {
        let BLOCK_SIZE = 16 * 8;
        const left_matrix_rows = matrix_size;
        const right_matrix_cols = matrix_size;
        const matrices_inner_dim = matrix_size;
		let num_timestamps = 2;
		
		const [left_matrix, gpu_buf_left_matrix] = CreateRandSquareMatrix(device, matrix_size);
		const [right_matrix, gpu_buf_right_matrix] = CreateRandSquareMatrix(device, matrix_size);
		const [result_matrix_bytes, gpu_buf_result_matrix_write] = CreateResultMatrix(device, matrix_size);

        const commandEncoder = device.createCommandEncoder();
		
		const [timestamp_query_set, timestamp_command] = SetUpTimestampQuery(device);
		var passEncoder = commandEncoder.beginComputePass({ timestampWrites: timestamp_command });
		
        const shader_module = await LoadShader(device);
		const computePipeline = CreatePipeline(device, shader_module);
        passEncoder.setPipeline(computePipeline);
		
		BindDataBuffers(device, passEncoder, computePipeline, gpu_buf_left_matrix, gpu_buf_right_matrix, gpu_buf_result_matrix_write);
		
		SetUpWorkgroups(passEncoder, left_matrix_rows, right_matrix_cols, BLOCK_SIZE, repetitions)
		
        passEncoder.end();
		
        const timestamp_bytes = num_timestamps * BigInt64Array.BYTES_PER_ELEMENT;
        const gpu_buf_timestamp_write = device.createBuffer({
            size: timestamp_bytes,
            usage: GPUBufferUsage.QUERY_RESOLVE |
                GPUBufferUsage.COPY_SRC,
        });
        commandEncoder.resolveQuerySet(timestamp_query_set, 0, 2, gpu_buf_timestamp_write, 0);

        const gpu_buf_result_matrix_read = GPUCopyResultsToRead(device, commandEncoder, gpu_buf_result_matrix_write, result_matrix_bytes);
        const gpu_buf_timestamp_read = GPUCopyResultsToRead(device, commandEncoder, gpu_buf_timestamp_write, timestamp_bytes);

        const gpuCommand = commandEncoder.finish();
		
        let js_start_time = Date.now();
        device.queue.submit([gpuCommand]);
		
		// Calc gpu performance numbers
		const num_float_ops = repetitions * 2 * left_matrix_rows * right_matrix_cols * matrices_inner_dim;
        const gpu_ns = await ReadGPUTimeDelta(gpu_buf_timestamp_read);
		const avg_gpu_ms = gpu_ns / 1e6 / repetitions;
		const gpu_gflopts = num_float_ops / gpu_ns;
		
        await gpu_buf_result_matrix_read.mapAsync(GPUMapMode.READ);
        let js_end_time = Date.now();
		
		// Calc js performance numbers
		const js_ms = Number(js_end_time - js_start_time);
		const avg_js_ms = js_ms / repetitions;
		const js_ns = js_ms*1e6;
		const js_gflopts = num_float_ops / js_ns;
		
		// get the result matrix
		const mapped_range_result_matrix = gpu_buf_result_matrix_read.getMappedRange();
        const result_matrix_array = new Float32Array(mapped_range_result_matrix);		
        const [verification_result, verification_ms] = Verify(left_matrix, right_matrix, result_matrix_array);
		
		// Display result
		let info = "No adapter info";
		if (adapter.info)
			info = `${adapter.info.vendor} ${adapter.info.architecture}`;
		
		const browser_info = navigator.userAgent;
		const resultStr = `
	         browser_info : ${browser_info}
	            algorithm : optimized gpu matrix multiplication using BLOCK_SIZE = 16 and TILE = 8x8
	             gpu info : ${info}
	                 work : 32bit float matrix multiplication ${left_matrix_rows}x${matrices_inner_dim} X ${matrices_inner_dim}x${right_matrix_cols} 
	          repetitions : ${repetitions}
	 js time / repetition : ${avg_js_ms} ms
	gpu time / repetition : ${avg_gpu_ms} ms
	            js GFLOPS : ${js_gflopts}
	           gpu GFLOPS : ${gpu_gflopts}
	  verification result : ${verification_result}
	    verification time : ${verification_ms} ms
    `;
        DisplayResult(resultStr);
		
    };
	
})();


function Verify(left_matrix, right_matrix, gpu_result_matrix_array)
{
	let start_time = Date.now();
	const left_matrix_rows = left_matrix[0];
	const right_matrix_cols = right_matrix[1];
	const matrices_inner_dim = left_matrix[1];
	const cpu_result_matrix_array = new Float32Array(left_matrix_rows * right_matrix_cols);
	let num_timestamps = 2;
	CPUMatrixMult(left_matrix_rows, right_matrix_cols, matrices_inner_dim, left_matrix.slice(num_timestamps), right_matrix.slice(num_timestamps), cpu_result_matrix_array);

	for (var left_matrix_row = 0; left_matrix_row < left_matrix_rows; left_matrix_row++) {
		for (var right_matrix_col = 0; right_matrix_col < right_matrix_cols; right_matrix_col++) {
			const result_matrix_index = left_matrix_row * right_matrix_cols + right_matrix_col;
			const gpu_val = gpu_result_matrix_array[num_timestamps+result_matrix_index];
			const cpu_val = cpu_result_matrix_array[result_matrix_index];
			if (Math.abs(cpu_val - gpu_val) > 1e-2) {
				const verification_ms = Number(Date.now() - start_time);
				return [`Error at ${left_matrix_row} ${right_matrix_col} cpu_val=${cpu_val} gpu_val=${gpu_val}`, verification_ms];
			}
		}
	}

	const verification_ms = Number(Date.now() - start_time);
	return ["Verification passed", verification_ms];
};
	
function CPUMatrixMult(left_matrix_rows, right_matrix_cols, matrices_inner_dim, left_matrix, right_matrix, result_matrix_array) {
    for (var left_matrix_row = 0; left_matrix_row < left_matrix_rows; left_matrix_row++) {
        for (let inner_index = 0; inner_index < matrices_inner_dim; inner_index++) {
            const left_matrix_val = left_matrix[left_matrix_row * matrices_inner_dim + inner_index];
            for (var right_matrix_col = 0; right_matrix_col < right_matrix_cols; right_matrix_col++) {
				const result_matrix_index = left_matrix_row * right_matrix_cols + right_matrix_col;
				const right_matrix_val = right_matrix[inner_index * right_matrix_cols + right_matrix_col];
                result_matrix_array[result_matrix_index] += left_matrix_val * right_matrix_val;
            }
        }
    }
}
