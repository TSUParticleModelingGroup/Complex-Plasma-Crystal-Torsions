void Display()
{
	drawPicture();
}

void idle()
{
	n_body();
}

void reshape(int w, int h)
{
	glViewport(0, 0, (GLsizei) w, (GLsizei) h);
}

void frustrumView()
{
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glFrustum(-0.2, 0.2, -0.2, 0.2, Near, Far);
	glMatrixMode(GL_MODELVIEW);
	FrustrumOrthoganal = 1;
}

void orthogonalView()
{
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-RadialConfinementScale, RadialConfinementScale, -RadialConfinementScale, RadialConfinementScale, Near, Far);
	glMatrixMode(GL_MODELVIEW);
	FrustrumOrthoganal = 0;
}

void topView()
{
	glLoadIdentity();
	glTranslatef(0.0, -CenterY, 0.0);
	glTranslatef(0.0, 0.0, -RadialConfinementScale);
	glTranslatef(CenterOfView.x, CenterOfView.y, CenterOfView.z);
	glRotatef(90.0, 1.0, 0.0, 0.0);
	glTranslatef(-CenterOfView.x, -CenterOfView.y, -CenterOfView.z);
	View = 1;
}

void sideView()
{
	glLoadIdentity();
	glTranslatef(0.0, -CenterY, 0.0);
	glTranslatef(0.0, 0.0, -RadialConfinementScale);
	//glRotatef(-90.0, 1.0, 0.0, 0.0);
	View = 0;
}

void centeredTopView()
{
	//glTranslatef(-CenterOfView.x, -CenterOfView.y, -CenterOfView.z);
	//CenterOfView.x = 0.0;
	//CenterOfView.y = 0.0;
	//CenterOfView.z = 0.0;
		
	glLoadIdentity();
	glTranslatef(0.0, -CenterY, 0.0);
	glTranslatef(0.0, 0.0, -RadialConfinementScale);
	CenterOfView.y = CenterY;
	CenterOfView.z = CenterZ;
	glTranslatef(CenterOfView.x, CenterOfView.y, CenterOfView.z);
	glRotatef(90.0, 1.0, 0.0, 0.0);
	glTranslatef(-CenterOfView.x, -CenterOfView.y, -CenterOfView.z);
	drawPicture();
}

string getTimeStamp() // Returns a timestamp in M-D-Y-H.M.S format.
{
	// Want to get a time stamp string representing current date/time, so we have a
	// unique name for each video/screenshot taken.
	time_t t = time(0); 
	struct tm * now = localtime( & t );
	int month = now->tm_mon + 1, day = now->tm_mday, year = now->tm_year, 
				curTimeHour = now->tm_hour, curTimeMin = now->tm_min, curTimeSec = now->tm_sec;
	stringstream smonth, sday, syear, stimeHour, stimeMin, stimeSec;
	smonth << month;
	sday << day;
	syear << (year + 1900); // The computer starts counting from the year 1900, so 1900 is year 0. So we fix that.
	stimeHour << curTimeHour;
	stimeMin << curTimeMin;
	stimeSec << curTimeSec;
	string timeStamp;

	if (curTimeMin <= 9)
	{	
		timeStamp = smonth.str() + "-" + sday.str() + "-" + syear.str() + '_' + stimeHour.str() + ".0" + stimeMin.str() + "." + stimeSec.str();
	}
	else
	{			
		timeStamp = smonth.str() + "-" + sday.str() + '-' + syear.str() + "_" + stimeHour.str() + "." + stimeMin.str() + "." + stimeSec.str();
	}
	return timeStamp;
}

void KeyPressed(unsigned char key, int x, int y)
{
	// 0.2 was original value for dx, dy, and dz.
	float dx = 0.05f / 2.0f; 
	float dy = 0.05f / 2.0f;
	float dz = 0.05f / 2.0f;

	if(key == 'q')
	{
		glutDestroyWindow(Window);
		printf("\nExiting....\n\nGood Bye\n");
		exit(0);
	}
	
	if(key == '!')
	{
		//glTranslatef(-CenterOfView.x, -CenterOfView.y, -CenterOfView.z);
		//CenterOfView.x = 0.0;
		//CenterOfView.y = 0.0;
		//CenterOfView.z = 0.0;
		
		glLoadIdentity();
		glTranslatef(0.0, -CenterY, 0.0);
		glTranslatef(0.0, 0.0, -RadialConfinementScale);
		CenterOfView.y = CenterY;
		CenterOfView.z = CenterZ;
		glTranslatef(CenterOfView.x, CenterOfView.y, CenterOfView.z);
		glRotatef(90.0, 1.0, 0.0, 0.0);
		glTranslatef(-CenterOfView.x, -CenterOfView.y, -CenterOfView.z);
		drawPicture();
	}
	
	/*
	void standardTopView()
	{
	      glLoadIdentity();
	      glTranslatef(0.0, -CenterY, 0.0);
	      glTranslatef(0.0, 0.0, -RadialConfinementScale);
	      CenterOfView.y = CenterY;
	      CenterOfView.z = CenterZ;
	      glTranslatef(CenterOfView.x, CenterOfView.y, CenterOfView.z);
	      glRotatef(90.0, 1.0, 0.0, 0.0);
	      glTranslatef(-CenterOfView.x, -CenterOfView.y, -CenterOfView.z);
	      drawPicture();
	}

	void topView()
	{
	      // First set to side view.
	      glLoadIdentity();
	      glTranslatef(0.0, -CenterY, 0.0);
	      glTranslatef(0.0, 0.0, -RadialConfinementScale);
	      // Now move over
	      glTranslatef(CenterOfView.x, CenterOfView.y, CenterOfView.z);
	      // Rotate to look down.
	      glRotatef(90.0, 1.0, 0.0, 0.0);
	      // Move back
	      glTranslatef(-CenterOfView.x, -CenterOfView.y, -CenterOfView.z);
	      drawPicture();
	}
*/
	
	// This a magic key we used to place dust to make a movie
	if(key == '2')
	{
		// Pull dust grain information down from the GPU.
		cudaMemcpy( DustPositionCPU, DustPositionGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
		errorCheck("cudaMemcpy DustPositionCPU down");
		
		View = 0;
		sideView();
		Pause = 0;
		Freeze = 1;
		FrustrumOrthoganal = 1;
		frustrumView();
		RadialConfinementViewingAids = 0;
		DustViewingAids = 1;
		XYZAdjustments = 1;
		BaseIonWakeFractionalCharge = 0.5;
		SelectedDustGrainId = 2;
		
		drawPicture();
		terminalPrint();
		
		DustPositionCPU[1].x = 0.0;
		DustPositionCPU[1].y = RadialConfinementHeight;
		DustPositionCPU[1].z = 0.0;
		
		DustPositionCPU[2].x = 0.0;
		DustPositionCPU[2].y = RadialConfinementHeight - 0.5;
		DustPositionCPU[2].z = 0.0;
		
		glTranslatef(0.0, -RadialConfinementHeight/2.0, (RadialConfinementScale - 3.0));
		CenterOfView.x += 0.0;
		CenterOfView.y += -RadialConfinementHeight/2.0;
		CenterOfView.z += (RadialConfinementScale - 3.0);
		drawPicture();

		// Push modifications back up to the GPU.
		cudaMemcpy( DustPositionGPU, DustPositionCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
		errorCheck("cudaMemcpy DustPositionCPU up");
	}
	
	if(key == '3')
	{
		// Pull dust grain information down from the GPU.
		cudaMemcpy( DustPositionCPU, DustPositionGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
		errorCheck("cudaMemcpy DustPositionCPU down");
		
		View = 0;
		sideView();
		Pause = 0;
		Freeze = 1;
		FrustrumOrthoganal = 1;
		frustrumView();
		RadialConfinementViewingAids = 0;
		DustViewingAids = 1;
		XYZAdjustments = 1;
		BaseIonWakeFractionalCharge = 0.5;
		SelectedDustGrainId = 3;
		
		drawPicture();
		terminalPrint();
		
		DustPositionCPU[1].x = -0.2;
		DustPositionCPU[1].y = RadialConfinementHeight;
		DustPositionCPU[1].z = 0.0;
		
		DustPositionCPU[2].x = 0.2;
		DustPositionCPU[2].y = RadialConfinementHeight -0.3;
		DustPositionCPU[2].z = 0.0;
		
		DustPositionCPU[3].x = 0.0;
		DustPositionCPU[3].y = RadialConfinementHeight - 1.0;
		DustPositionCPU[3].z = 0.0;

		glTranslatef(0.0, -RadialConfinementHeight/2.0, (RadialConfinementScale - 3.0));
		CenterOfView.x += 0.0;
		CenterOfView.y += -RadialConfinementHeight/2.0;
		CenterOfView.z += (RadialConfinementScale - 3.0);
		drawPicture();
		
		// Push modifications back up to the GPU.
		cudaMemcpy( DustPositionGPU, DustPositionCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
		errorCheck("cudaMemcpy DustPositionCPU up");
	}
	
	
	if(key == '4')
	{
		// Pull dust grain information down from the GPU.
		cudaMemcpy( DustPositionCPU, DustPositionGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
		errorCheck("cudaMemcpy DustPositionCPU down");
		
		View = 0;
		sideView();
		Pause = 0;
		Freeze = 1;
		FrustrumOrthoganal = 1;
		frustrumView();
		RadialConfinementViewingAids = 0;
		DustViewingAids = 1;
		XYZAdjustments = 1;
		BaseIonWakeFractionalCharge = 0.5;
		SelectedDustGrainId = 4;
		
		drawPicture();
		terminalPrint();
		
		DustPositionCPU[1].x = -0.3;
		DustPositionCPU[1].y = RadialConfinementHeight;
		DustPositionCPU[1].z = 0.0;
		
		DustPositionCPU[2].x = 0.3;
		DustPositionCPU[2].y = RadialConfinementHeight;
		DustPositionCPU[2].z = 0.0;
		
		DustPositionCPU[3].x = 0.2;
		DustPositionCPU[3].y = RadialConfinementHeight - 0.5;
		DustPositionCPU[3].z = 0.0;
		
		DustPositionCPU[4].x = 2.0;
		DustPositionCPU[4].y = RadialConfinementHeight - 1.0;
		DustPositionCPU[4].z = 0.0;

		glTranslatef(0.0, -RadialConfinementHeight/2.0, (RadialConfinementScale - 3.0));
		CenterOfView.x += 0.0;
		CenterOfView.y += -RadialConfinementHeight/2.0;
		CenterOfView.z += (RadialConfinementScale - 3.0);
		drawPicture();
		
		// Push modifications back up to the GPU.
		cudaMemcpy( DustPositionGPU, DustPositionCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
		errorCheck("cudaMemcpy DustPositionCPU up");
	}
	
	
	if(key == 'w') // Top view is 1, Side view is 0
	{
		if(View == 0) 
		{
			//View = 1;
			topView();
		}
		else 
		{
			//View = 0;
			sideView();
		}
		drawPicture();
		terminalPrint();
	}
	
	if(key == '!')
	{
		View = 1;
		centeredTopView();
		drawPicture();
		terminalPrint();
	}
	
	if(key == 'o') // Frustrum view is 1, Orthoganal view is 0
	{
		if(FrustrumOrthoganal == 0) 
		{
			//FrustrumOrthoganal = 1;
			frustrumView();
		}
		else 
		{
			//FrustrumOrthoganal = 0;
			orthogonalView();
		}
		drawPicture();
		terminalPrint();
	}
	
	if(key == 'e')
	{
		XYZAdjustments = (XYZAdjustments + 1)%4;
		terminalPrint();
	}
	
	// Control freezing of dust grain movement.
	// This just stops dust from moving all the other calculations are still being run.
	if (key == 'f')
	{
		if(Freeze == 0) Freeze = 1;
		else Freeze = 0;
		terminalPrint();
	}
	
	// Control the runing and pausing of the simulation.
	if (key == 'r')
	{
		if (Pause == 0) Pause = 1;
		else Pause = 0;
		terminalPrint();
	}
	
	// Turns tracers on and off
	if(key == 't')
	{
		if(Trace == 1) Trace = 0;
		else Trace = 1;
		drawPicture();
		terminalPrint();
	}
	
	// Turns radial confinement viewing adds on and off
	if(key == 'a')
	{
		if(RadialConfinementViewingAids == 1) RadialConfinementViewingAids = 0;
		else RadialConfinementViewingAids = 1;
		drawPicture();
		terminalPrint();
	}
	
	// Turns dust viewing adds on and off
	if (key == 'g')
	{
		if (DustViewingAids == 0) DustViewingAids = 1;
		else DustViewingAids = 0;
		drawPicture();
		terminalPrint();
	}
	
	if(key == 'm') // Movie Toggle
	{
		if(MovieOn == 0) // Start recording
		{
			MovieOn = 1;
			string ts = getTimeStamp();
			ts.append(".mp4");

			string y = "ffmpeg -loglevel quiet -r 60 -f rawvideo -pix_fmt rgba -s 1000x1000 -i - "
				      "-c:v libx264rgb -threads 0 -preset fast -y -pix_fmt yuv420p -crf 0 -vf vflip ";

			string z = y + ts;

			const char *ccx = z.c_str();

			MovieFile = popen(ccx, "w");
			printf("Movie successfully started recording.\n");
			//Buffer = new int[XWindowSize*YWindowSize];
			Buffer = (int*)malloc(XWindowSize*YWindowSize*sizeof(int));
			terminalPrint();
		}
		else // Stop recording
		{
			MovieOn = 0;
			pclose(MovieFile);
			printf("\nMovie was successfully completed.\n");
			free(Buffer);
			terminalPrint();
		}
	}
		
	if(key == 's') // Scream Shot
	{	
		int pauseFlag;
		FILE* ScreenShotFile;
		int* buffer;
		//const char* cmd = "ffmpeg -loglevel quiet -r 60 -f rawvideo -pix_fmt rgba -s 1000x1000 -i - "
		//              "-threads 0 -q:v 1 -qmin 1 -qmax 1  -src_range 0 -dst_range 1 -preset fast -y -pix_fmt yuv420p -crf 21 -vcodec libx264 -vf vflip output1.mp4";
		              
		const char* cmd = "ffmpeg -loglevel quiet -framerate 60 -f rawvideo -pix_fmt rgba -s 1000x1000 -i - "
						"-c:v libx264rgb -threads 0 -preset fast -y -crf 0 -vf vflip output1.mp4";
		ScreenShotFile = popen(cmd, "w");
		buffer = (int*)malloc(XWindowSize*YWindowSize*sizeof(int));
		
		if(Pause == 0) 
		{
			Pause = 1;
			pauseFlag = 0;
		}
		else
		{
			pauseFlag = 1;
		}
		
		for(int i = 0; i < 1; i++)
		{
			// drawPicture();
			glReadPixels(5, 5, XWindowSize, YWindowSize, GL_RGBA, GL_UNSIGNED_BYTE, buffer);
			fwrite(buffer, sizeof(int)*XWindowSize*YWindowSize, 1, ScreenShotFile);
		}
		
		pclose(ScreenShotFile);
		free(buffer);
		string ts = getTimeStamp(); // Only storing in a separate variable for debugging purposes.
		string s = "ffmpeg -loglevel quiet -i output1.mp4 -qscale:v 1 -qmin 1 -qmax 1 " + ts + ".jpeg";
		// Convert back to a C-style string.
		const char *ccx = s.c_str();
		system(ccx);
		system("rm output1.mp4");
		printf("\nScreenshot Captured: \n");
		cout << "Saved as " << ts << ".jpeg" << endl;
		Pause = pauseFlag;
		//ffmpeg -i output1.mp4 output_%03d.jpeg*/
	}
	
	// Adjusting power on lower plate: V increases power, v decreases power 
	double deltaBottomPlateForcingParameter = 1.0e7;
	if(key == 'V')
	{
		deltaBottomPlateForcingParameter = deltaBottomPlateForcingParameter*TimeUnit*TimeUnit*ChargeUnit/MassUnit;
		BottomPlateForcingParameter += deltaBottomPlateForcingParameter;
		terminalPrint();
	}
	if(key == 'v')
	{
		deltaBottomPlateForcingParameter = deltaBottomPlateForcingParameter*TimeUnit*TimeUnit*ChargeUnit/MassUnit;
		BottomPlateForcingParameter -= deltaBottomPlateForcingParameter;
		if(BottomPlateForcingParameter < 0.0) BottomPlateForcingParameter = 0.0;
		terminalPrint();
	}
	
	// Adjusting power on side of cavity: C increases power, c decreases power
	double deltaRadialConfinementStrength = 1.0e5;
	if(key == 'C')
	{
		deltaRadialConfinementStrength = deltaRadialConfinementStrength*TimeUnit*TimeUnit*ChargeUnit/MassUnit;
		RadialConfinementStrength += deltaRadialConfinementStrength;
		terminalPrint();
	}
	if(key == 'c')
	{
		deltaRadialConfinementStrength = deltaRadialConfinementStrength*TimeUnit*TimeUnit*ChargeUnit/MassUnit;
		RadialConfinementStrength -= deltaRadialConfinementStrength;
		if(RadialConfinementStrength < 0.0) RadialConfinementStrength = 0.0;
		terminalPrint();
	}
	
	// Adjusting pressure: P increases presure, p decreases presure
	double deltaPressure = 10000.0;
	if (key == 'P') // increase pressure
	{
		GasPressure += deltaPressure;
		//Drag = PressureConstant * GasPressure;
		terminalPrint();
	}
	if (key == 'p') // decrease pressure
	{
		GasPressure -= deltaPressure;
		if (GasPressure < 0.0) GasPressure = 0.0;
		//Drag = PressureConstant * GasPressure;
		terminalPrint();
	}
	
	//Adjusting point change
	if(key == 'i')
	{
		BaseIonWakeFractionalCharge -= 0.01;
		if(BaseIonWakeFractionalCharge < 0.0) BaseIonWakeFractionalCharge = 0.0;
	    	terminalPrint();
		drawPicture();
	}
	if(key == 'I')
	{
		BaseIonWakeFractionalCharge += 0.01;
	    	terminalPrint();
		drawPicture();
	}
	
	//Adjusting draw rate
	if (key == 'D')
	{
		DrawRate++;
		terminalPrint();
	}
	if (key == 'd')
	{
		if (DrawRate > 1) // Stops us from setting draw rate lower than 1.
		DrawRate--;
		terminalPrint();
	}

	if(XYZAdjustments == 0)
	{
		dx = 0.05f;
		dy = 0.05f;
		dz = 0.05f;
	
		if(key == 'x')
		{

			glTranslatef(-dx, 0.0, 0.0);
			CenterOfView.x += -dx;
			drawPicture();
		}
		if(key == 'X')
		{
			glTranslatef(dx, 0.0, 0.0);
			CenterOfView.x += dx;
			drawPicture();
		}
		
		if(key == 'y')
		{
			glTranslatef(0.0, -dy, 0.0);
			CenterOfView.y += -dy;
			drawPicture();
		}
		if(key == 'Y')
		{
			glTranslatef(0.0, dy, 0.0);
			CenterOfView.y += dy;
			drawPicture();
		}
		
		if(key == 'z')
		{
			glTranslatef(0.0, 0.0, -dz);
			CenterOfView.z += -dz;
			drawPicture();
		}
		if(key == 'Z')
		{
			glTranslatef(0.0, 0.0, dz);
			CenterOfView.z += dz;
			drawPicture();
		}
	}
	else if(XYZAdjustments == 1 && SelectedDustGrainId != -1)
	{
		dx = 0.05f;
		dy = 0.05f;
		dz = 0.05f;
		
		// Pull dust grain information down from the GPU.
		cudaMemcpy( DustPositionCPU, DustPositionGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
		errorCheck("cudaMemcpy DustPositionCPU down");

		if(key == 'x')
		{
			DustPositionCPU[SelectedDustGrainId].x += -dx;
			terminalPrint();
			drawPicture();
		}
		if(key == 'X')
		{
			DustPositionCPU[SelectedDustGrainId].x += dx;
			terminalPrint();
			drawPicture();
		}
		
		if(key == 'y')
		{
			DustPositionCPU[SelectedDustGrainId].y += -dy;
			terminalPrint();
			drawPicture();
		}
		if(key == 'Y')
		{
			DustPositionCPU[SelectedDustGrainId].y += dy;
			terminalPrint();
			drawPicture();
		}
		
		if(key == 'z')
		{
			DustPositionCPU[SelectedDustGrainId].z += -dz;
			terminalPrint();
			drawPicture();
		}
		if(key == 'Z')
		{
			DustPositionCPU[SelectedDustGrainId].z += dz;
			terminalPrint();
			drawPicture();
		}

		// Push modifications back up to the GPU.
		cudaMemcpy( DustPositionGPU, DustPositionCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
		errorCheck("cudaMemcpy DustPositionCPU up");
	}
	else if(XYZAdjustments == 2 && SelectedDustGrainId != -1)
	{
		dx = 0.05f;
		dy = 0.05f;
		dz = 0.05f;
		
		// Pull dust grain information down from the GPU.
		cudaMemcpy( DustVelocityCPU, DustVelocityGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
		errorCheck("cudaMemcpy DustVelocityCPU down");

		if(key == 'x')
		{
			DustVelocityCPU[SelectedDustGrainId].x += -dx;
			terminalPrint();
			drawPicture();
		}
		if(key == 'X')
		{
			DustVelocityCPU[SelectedDustGrainId].x += dx;
			terminalPrint();
			drawPicture();
		}
		
		if(key == 'y')
		{
			DustVelocityCPU[SelectedDustGrainId].y += -dy;
			terminalPrint();
			drawPicture();
		}
		if(key == 'Y')
		{
			DustVelocityCPU[SelectedDustGrainId].y += dy;
			terminalPrint();
			drawPicture();
		}
		
		if(key == 'z')
		{
			DustVelocityCPU[SelectedDustGrainId].z += -dz;
			terminalPrint();
			drawPicture();
		}
		if(key == 'Z')
		{
			DustVelocityCPU[SelectedDustGrainId].z += dz;
			terminalPrint();
			drawPicture();
		}
		
		// Push modifications back up to the GPU.
		cudaMemcpy( DustVelocityGPU, DustVelocityCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
		errorCheck("cudaMemcpy DustVelocityGPU up");
	}
	else if(XYZAdjustments == 3 && SelectedDustGrainId != -1)
	{
		dx = 0.0001f;
		
		// Pull dust grain information down from the GPU.
		cudaMemcpy( DustPositionCPU, DustPositionGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
		errorCheck("cudaMemcpy DustPositionCPU down");
		cudaMemcpy( DustVelocityCPU, DustVelocityGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
		errorCheck("cudaMemcpy DustVelocityCPU down");
		cudaMemcpy( DustForceCPU, DustForceGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
		errorCheck("cudaMemcpy DustVelocityCPU down");

		if(key == 'x')
		{
			DustVelocityCPU[SelectedDustGrainId].w += -dx;
			terminalPrint();
			drawPicture();
		}
		if(key == 'X')
		{
			DustVelocityCPU[SelectedDustGrainId].w += dx;
			terminalPrint();
			drawPicture();
		}
		
		// Get the diameter
		float diameter = DustVelocityCPU[SelectedDustGrainId].w;
		// Get the radius
		float radius = diameter/2.0;
			
		// Now the mass of this dust particle will be determined off of its diameter and its density (at present all the dust particles have the same density).
		float mass = DustDensity*(4.0/3.0)*PI*radius*radius*radius;
		
		// Setting the charge on the dust grain. Just using the base electrons per unit diameter.
		float numberOfElectrons = round(BaseElectronsPerUnitDiameter*diameter);
		float charge = numberOfElectrons*ElectronCharge;
		
		DustPositionCPU[SelectedDustGrainId].w = charge;
		DustVelocityCPU[SelectedDustGrainId].w = diameter;
		DustForceCPU[SelectedDustGrainId].w = mass;
		
		cudaMemcpy( DustPositionGPU, DustPositionCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
		errorCheck("cudaMemcpy DustPositionCPU up");
		cudaMemcpy( DustVelocityGPU, DustVelocityCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
		errorCheck("cudaMemcpy DustVelocityGPU up");
		cudaMemcpy( DustForceGPU, DustForceCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
		errorCheck("cudaMemcpy DustForceGPU up");
	}
	
	if (key == 'h') 
	{
		FILE *fptr; 
		char filename[100] = "help.txt", c;
		// Open file
		fptr = fopen(filename, "r");
		if (fptr == NULL)
		{
			printf("Cannot open help file (help.txt).\n");
			exit(0);
		}
		  
		// Read contents from file
		c = fgetc(fptr);
		while (c != EOF)
		{
			printf ("%c", c);
			c = fgetc(fptr);
		}
		 
		fclose(fptr);
		Pause = 1;	
	}
}

void mymouse(int button, int state, int x, int y)
{	
	float myX, myZ;
	//float dustX, dustY, dustZ;
	float closest;
	float dx, dz, d;
	//double diameter, radius, mass, charge, electronsPerUnitDiameter;
	
	if(state == GLUT_DOWN)
	{
		if(DustSelectionOn == 1)
		{
			// You need to copy everything down from the GPU because the mass is in force.w the diameter is in vel.w and the charge is in pos.w
			cudaMemcpy( DustPositionCPU, DustPositionGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
			errorCheck("cudaMemcpy DustPositionCPU down");
			
			if(button == GLUT_LEFT_BUTTON)
			{
				// Setting the previously selected dust to Magenta.
				if(SelectedDustGrainId != -1)
				{
					DustColor[SelectedDustGrainId].x = 1.0;
					DustColor[SelectedDustGrainId].y = 0.0;
					DustColor[SelectedDustGrainId].z = 1.0;
				}
				
				myX =  ( 2.0*x/XWindowSize - 1.0)*RadialConfinementScale;
				myZ = -(-2.0*y/YWindowSize + 1.0)*RadialConfinementScale;
				
				// Flashing a big yellow ball where you selected.
				glColor3d(1.0, 1.0, 0.0);
					glPushMatrix();
					glTranslatef(myX, RadialConfinementHeight/2.0, myZ);
					glutSolidSphere(0.5,20,20);
				glPopMatrix();
				glutSwapBuffers();
				usleep(200000);
				
				// Finding the closest dust grain to where your mouse clicked.
				closest = 10000000.0;
				SelectedDustGrainId = -1;
				for(int i = 1; i < NumberOfDustParticles; i++)
				{
					dx = DustPositionCPU[i].x - myX;
					dz = DustPositionCPU[i].z - myZ;
					d = sqrt(dx*dx + dz*dz);
					if(d < closest)
					{
						closest = d;
						SelectedDustGrainId = i;
					}
				}
				
				// Setting the color of the selected dust to blue.
				if(SelectedDustGrainId != -1)
				{
					DustColor[SelectedDustGrainId].x = 0.0;
					DustColor[SelectedDustGrainId].y = 0.0;
					DustColor[SelectedDustGrainId].z = 1.0;
				}
				else
				{
					printf("\n Error no dust grain selected\n");
					exit(0);
				}
				drawPicture();
				terminalPrint();
				DustSelectionOn = 0;
			}
		}
	}
}
