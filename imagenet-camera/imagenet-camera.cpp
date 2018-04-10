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

//#include "gstpipeline.h"
#include "gstPipeline.h"
#include "glDisplay.h"
#include "glTexture.h"

#include <stdio.h>
#include <signal.h>
#include <unistd.h>

#include "cudaNormalize.h"
#include "cudaFont.h"
#include "imageNet.h"



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
    printf("imagenet-pipeline\n  args (%i):  ", argc);

	for( int i=0; i < argc; i++ )
		printf("%i [%s]  ", i, argv[i]);
		
	printf("\n\n");
	

	/*
	 * attach signal handler
	 */
	if( signal(SIGINT, sig_handler) == SIG_ERR )
		printf("\ncan't catch SIGINT\n");


	/*
     * create the pipeline device
	 */
   /* std::ostringstream ss;
    ss << "rtspsrc location=rtsp://root:root@192.168.1.90/axis-media/media.amp?resolution=1280x720  drop-on-latency=0 latency=100 ! ";
    ss << "queue max-size-buffers=200 max-size-time=1000000000  max-size-bytes=10485760 min-threshold-time=10 ! ";
    ss << "rtph264depay ! h264parse ! omxh264dec ! video/x-raw, format=(string)NV12 ! ";
    //ss << "nvvidconv flip-method=2 ! video/x-raw(memory:NVMM), width=(int)1920, height=(int)720, format=RGB ! ";
    ss << "appsink name=mysink";
    GstPipeline* pipeline = GstPipeline::Create(ss, 1920, 720, 12);
*/

   std::ostringstream ss;
   ss << "rtspsrc location=rtsp://root:root@192.168.1.90/axis-media/media.amp?resolution=1280x720  drop-on-latency=0 latency=100 ! ";
   ss << "queue max-size-buffers=200 max-size-time=1000000000  max-size-bytes=10485760 min-threshold-time=10 ! ";
   ss << "rtph264depay ! h264parse ! omxh264dec ! ";
  // ss << "nvvidconv flip-method=2 ! video/x-raw(memory:NVMM), width=(int)1920, height=(int)720, format=NV12 ! ";
   //ss << "nvvidconv flip-method=2 ! ";
   ss << "appsink name=mysink";

  // ss << "queue ! rtph264depay ! queue ! h264parse ! queue ! omxh264dec ! nvvidconv ! video/x-raw, format=NV12 ! queue ! appsink name=mysink";

   gstPipeline* pipeline = gstPipeline::Create(ss.str(), 1280, 720, 12 );
    //gstpipeline* pipeline = gstpipeline::Create(DEFAULT_pipeline);
	
    if( !pipeline )
	{
        printf("\nimagenet-pipeline:  failed to initialize video device\n");
		return 0;
	}
	
    printf("\nimagenet-pipeline:  successfully initialized video device\n");
    printf("    width:  %u\n", pipeline->GetWidth());
    printf("   height:  %u\n", pipeline->GetHeight());
    printf("    depth:  %u (bpp)\n\n", pipeline->GetPixelDepth());
	

	/*
	 * create imageNet
	 */
	imageNet* net = imageNet::Create(argc, argv);
	
	if( !net )
	{
		printf("imagenet-console:   failed to initialize imageNet\n");
		return 0;
	}


	/*
	 * create openGL window
	 */
	glDisplay* display = glDisplay::Create();
	glTexture* texture = NULL;
	
	if( !display ) {
        printf("\nimagenet-pipeline:  failed to create openGL display\n");
	}
	else
	{
        texture = glTexture::Create(pipeline->GetWidth(), pipeline->GetHeight(), GL_RGBA32F_ARB/*GL_RGBA8*/);

		if( !texture )
            printf("imagenet-pipeline:  failed to create openGL texture\n");
	}
	
	
	/*
	 * create font
	 */
	cudaFont* font = cudaFont::Create();
	

	/*
	 * start streaming
	 */
    if( !pipeline->Open() )
	{
        printf("\nimagenet-pipeline:  failed to open pipeline for streaming\n");
		return 0;
	}
	
    printf("\nimagenet-pipeline:  pipeline open for streaming\n");
	
	
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
            printf("\nimagenet-pipeline:  failed to capture frame\n");
		//else
        //	printf("imagenet-pipeline:  recieved new frame  CPU=0x%p  GPU=0x%p\n", imgCPU, imgCUDA);
		
		// convert from YUV to RGBA
		void* imgRGBA = NULL;
		
        if( !pipeline->ConvertRGBA(imgCUDA, &imgRGBA) )
            printf("imagenet-pipeline:  failed to convert from NV12 to RGBA\n");

		// classify image
        const int img_class = net->Classify((float*)imgRGBA, pipeline->GetWidth(), pipeline->GetHeight(), &confidence);
	
		if( img_class >= 0 )
		{
            printf("imagenet-pipeline:  %2.5f%% class #%i (%s)\n", confidence * 100.0f, img_class, net->GetClassDesc(img_class));

			if( font != NULL )
			{
				char str[256];
				sprintf(str, "%05.2f%% %s", confidence * 100.0f, net->GetClassDesc(img_class));
	
                font->RenderOverlay((float4*)imgRGBA, (float4*)imgRGBA, pipeline->GetWidth(), pipeline->GetHeight(),
								    str, 0, 0, make_float4(255.0f, 255.0f, 255.0f, 255.0f));
			}
			
			if( display != NULL )
			{
				char str[256];
				sprintf(str, "TensorRT build %x | %s | %s | %04.1f FPS", NV_GIE_VERSION, net->GetNetworkName(), net->HasFP16() ? "FP16" : "FP32", display->GetFPS());
				//sprintf(str, "TensorRT build %x | %s | %04.1f FPS | %05.2f%% %s", NV_GIE_VERSION, net->GetNetworkName(), display->GetFPS(), confidence * 100.0f, net->GetClassDesc(img_class));
				display->SetTitle(str);	
			}	
		}	


		// update display
		if( display != NULL )
		{
			display->UserEvents();
			display->BeginRender();

			if( texture != NULL )
			{
				// rescale image pixel intensities for display
				CUDA(cudaNormalizeRGBA((float4*)imgRGBA, make_float2(0.0f, 255.0f), 
								   (float4*)imgRGBA, make_float2(0.0f, 1.0f), 
                                   pipeline->GetWidth(), pipeline->GetHeight()));

				// map from CUDA to openGL using GL interop
				void* tex_map = texture->MapCUDA();

				if( tex_map != NULL )
				{
					cudaMemcpy(tex_map, imgRGBA, texture->GetSize(), cudaMemcpyDeviceToDevice);
					texture->Unmap();
				}

				// draw the texture
				texture->Render(100,100);		
			}

			display->EndRender();
		}
	}
	
    printf("\nimagenet-pipeline:  un-initializing video device\n");
	
	
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
	
    printf("imagenet-pipeline:  video device has been un-initialized.\n");
    printf("imagenet-pipeline:  this concludes the test of the video device.\n");
	return 0;
}

