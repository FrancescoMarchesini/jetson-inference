/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "gstPipeline.h"

#include "glDisplay.h"
#include "glTexture.h"

#include <stdio.h>
#include <signal.h>
#include <unistd.h>

#include "cudaMappedMemory.h"
#include "cudaNormalize.h"
#include "cudaFont.h"

#include "segNet.h"


#define DEFAULT_pipeline -1	// -1 for onboard pipeline, or change to index of /dev/video V4L2 pipeline (>=0)
		

bool signal_recieved = false;

void sig_handler(int signo)
{
	if( signo == SIGINT )
	{
		printf("received SIGINT\n");
		signal_recieved = true;
	}
}


int main( int argc, char** argv )
{
    printf("segnet-pipeline\n  args (%i):  ", argc);

	for( int i=0; i < argc; i++ )
		printf("%i [%s]  ", i, argv[i]);
		
	printf("\n\n");
	
	
	if( signal(SIGINT, sig_handler) == SIG_ERR )
		printf("\ncan't catch SIGINT\n");


	/*
     * create the pipeline device
	 */
    std::ostringstream ss;
    uint32_t w, h;
    std::stringstream ssw(argv[3]);
    std::stringstream ssh(argv[4]);
    ssw >> w;
    ssh >> h;

    ss << "rtspsrc location=rtsp://"<<argv[2]<<"/axis-media/media.amp?resolution="<<argv[3]<<"x"<<argv[4]<<"  drop-on-latency=1 latency=10 ! ";
    ss << "queue max-size-buffers=200 max-size-time=1000000000  max-size-bytes=10485760 min-threshold-time=10 ! ";
    ss << "rtph264depay ! h264parse ! omxh264dec ! ";
    ss << "video/x-raw, width=(int)"<< w <<", height=(int)"<< h <<", format=(string)NV12 ! ";
    ss << "appsink name=mysink";

     gstPipeline* pipeline = gstPipeline::Create( ss.str(), w, h, 12 );
    if( !pipeline )
	{
        printf("\nsegnet-pipeline:  failed to initialize video device\n");
		return 0;
	}
	
    printf("\nsegnet-pipeline:  successfully initialized video device\n");
    printf("    width:  %u\n", pipeline->GetWidth());
    printf("   height:  %u\n", pipeline->GetHeight());
    printf("    depth:  %u (bpp)\n\n", pipeline->GetPixelDepth());
	

	/*
	 * create segNet
	 */
	segNet* net = segNet::Create(argc, argv);
	
	if( !net )
	{
        printf("segnet-pipeline:   failed to initialize imageNet\n");
		return 0;
	}

	// set alpha blending value for classes that don't explicitly already have an alpha	
	net->SetGlobalAlpha(120);

	// allocate segmentation overlay output buffer
	float* outCPU  = NULL;
	float* outCUDA = NULL;

    if( !cudaAllocMapped((void**)&outCPU, (void**)&outCUDA, pipeline->GetWidth() * pipeline->GetHeight() * sizeof(float) * 4) )
	{
        printf("segnet-pipeline:  failed to allocate CUDA memory for output image (%ux%u)\n", pipeline->GetWidth(), pipeline->GetHeight());
		return 0;
	}

	
	/*
	 * create openGL window
	 */
	glDisplay* display = glDisplay::Create();
	glTexture* texture = NULL;
	
	if( !display ) {
        printf("\nsegnet-pipeline:  failed to create openGL display\n");
	}
	else
	{
        texture = glTexture::Create(pipeline->GetWidth(), pipeline->GetHeight(), GL_RGBA32F_ARB/*GL_RGBA8*/);

		if( !texture )
            printf("segnet-pipeline:  failed to create openGL texture\n");
	}
	

	/*
	 * start streaming
	 */
    if( !pipeline->Open() )
	{
        printf("\nsegnet-pipeline:  failed to open pipeline for streaming\n");
		return 0;
	}
	
    printf("\nsegnet-pipeline:  pipeline open for streaming\n");
	
	
	/*
	 * processing loop
	 */
	float confidence = 0.0f;
	
	while( !signal_recieved )
	{
		void* imgCPU  = NULL;
		void* imgCUDA = NULL;
		
		// get the latest frame
        if( !pipeline->Capture(&imgCPU, &imgCUDA, 1000) )
            printf("\nsegnet-pipeline:  failed to capture frame\n");

		// convert from YUV to RGBA
		void* imgRGBA = NULL;
		
        if( !pipeline->ConvertRGBA(imgCUDA, &imgRGBA) )
            printf("segnet-pipeline:  failed to convert from NV12 to RGBA\n");

        // process image
        if( !net->Overlay((float*)imgRGBA, (float*)outCUDA, pipeline->GetWidth(), pipeline->GetHeight()))
		{
			printf("segnet-console:  failed to process segmentation overlay.\n");
			continue;
		}

		// update display
		if( display != NULL )
		{
			char str[256];
			sprintf(str, "TensorRT build %x | %s | %04.1f FPS", NV_GIE_VERSION, net->HasFP16() ? "FP16" : "FP32", display->GetFPS());
			//sprintf(str, "GIE build %x | %s | %04.1f FPS | %05.2f%% %s", NV_GIE_VERSION, net->GetNetworkName(), display->GetFPS(), confidence * 100.0f, net->GetClassDesc(img_class));
			display->SetTitle(str);	
	
			// next frame in the window
			display->UserEvents();
			display->BeginRender();

			if( texture != NULL )
			{
				// rescale image pixel intensities for display
				CUDA(cudaNormalizeRGBA((float4*)outCUDA, make_float2(0.0f, 255.0f), 
								   (float4*)outCUDA, make_float2(0.0f, 1.0f), 
                                   pipeline->GetWidth(), pipeline->GetHeight()));

				// map from CUDA to openGL using GL interop
				void* tex_map = texture->MapCUDA();

				if( tex_map != NULL )
				{
					cudaMemcpy(tex_map, outCUDA, texture->GetSize(), cudaMemcpyDeviceToDevice);
					texture->Unmap();
				}

				// draw the texture
				texture->Render(10, 10);		
			}

			display->EndRender();
        }
	}
	
    printf("\nsegnet-pipeline:  un-initializing video device\n");
	
	
	/*
     * shutdown the pipeline device
	 */
    if( pipeline != NULL )
	{
        delete pipeline;
        pipeline = NULL;
	}

	if( display != NULL )
	{
		delete display;
		display = NULL;
	}
	
    printf("segnet-pipeline:  video device has been un-initialized.\n");
    printf("segnet-pipeline:  this concludes the use of the device.\n");
	return 0;
}

