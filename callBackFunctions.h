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
	drawPicture();
}

void orthogonalView()
{
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-RadialConfinementScale, RadialConfinementScale, -RadialConfinementScale, RadialConfinementScale, Near, Far);
	glMatrixMode(GL_MODELVIEW);
	drawPicture();
}

void topView()
{
	glLoadIdentity();
	glTranslatef(0.0, -CenterY, 0.0);
	glTranslatef(0.0, 0.0, -RadialConfinementScale);
	glTranslatef(CenterOfView.x, CenterOfView.y, CenterOfView.z);
	glRotatef(90.0, 1.0, 0.0, 0.0);
	glTranslatef(-CenterOfView.x, -CenterOfView.y, -CenterOfView.z);
	drawPicture();
}

void sideView()
{
	glLoadIdentity();
	glTranslatef(0.0, -CenterY, 0.0);
	glTranslatef(0.0, 0.0, -RadialConfinementScale);
	drawPicture();
}

/*
	Returns a timestamp in M-D-Y-H.M.S format.
*/
string getTimeStamp()
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
		timeStamp = smonth.str() + "-" + sday.str() + "-" + syear.str() + '_' + stimeHour.str() + ".0" + stimeMin.str() + 
					"." + stimeSec.str();
	else			
		timeStamp = smonth.str() + "-" + sday.str() + '-' + syear.str() + "_" + stimeHour.str() + "." + stimeMin.str() +
					"." + stimeSec.str();
	return timeStamp;
}

void KeyPressed(unsigned char key, int x, int y)
{
	// 0.2 was original value for dx, dy, and dz.
	float dx = 0.05f / 2.0f; 
	float dy = 0.05f / 2.0f;
	float dz = 0.05f / 2.0f;
	// double diameter, radius, mass, charge, electronsPerUnitDiameter, numberOfElectrons;

	if(key == 'q')
	{
		glutDestroyWindow(Window);
		printf("\nExiting....\n\nGood Bye\n");
		exit(0);
	}
	
	if(key == 'w') // Top view is 1, Side view is 0
	{
		if(View == 0) 
		{
			View = 1;
			topView();
		}
		else 
		{
			View = 0;
			sideView();
		}
		terminalPrint();
	}
	
	// Control freezing of dust grains.
	if (key == 'f')
	{
		if(Freeze == 0) Freeze = 1;
		else Freeze = 0;
		terminalPrint();
	}
	
	if (key == 'r')
	{
		//drawPicture();
		if (Pause == 0) Pause = 1;
		else Pause = 0;
		//drawPicture();
		terminalPrint();
		//drawPicture();
	}
	if(key == 't')
	{
		if(Trace == 1) Trace = 0;
		else Trace = 1;
		terminalPrint();
	}
	if(key == 'a')
	{
		if(RadialConfinementViewingAids == 1) RadialConfinementViewingAids = 0;
		else RadialConfinementViewingAids = 1;
		drawPicture();
		terminalPrint();
	}
	if (key == 'g')
	{
		if (DustViewingAids == 0) DustViewingAids = 1;
		else DustViewingAids = 0;
		terminalPrint();
	}
	
	
	
	if (key == '*')
	{
		Pause = 1;
		// float dx,dy,dz,d;
		//cudaMemcpy( DustParentChildCPU, DustParentChildGPU, NumberOfDustParticles*sizeof(dustParentChildStructure), cudaMemcpyDeviceToHost);
		cudaMemcpy( DustPositionCPU, DustPositionGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
		for(int i = 0; i < NumberOfDustParticles; i++)
		{
			printf("id = %d, x = %f, y = %f, z = %f\n", i, DustPositionCPU[i].x, DustPositionCPU[i].y, DustPositionCPU[i].z);
			/*if(DustParentChildCPU[i].childId[0] != -1)
			{
				dx = DustPositionCPU[i].x - DustPositionCPU[DustParentChildCPU[i].childId[0]].x;
				dy = DustPositionCPU[i].y - DustPositionCPU[DustParentChildCPU[i].childId[0]].y;
				dz = DustPositionCPU[i].z - DustPositionCPU[DustParentChildCPU[i].childId[0]].z;
				d = sqrt(dx*dx + dy*dy + dz*dz);
				printf(" Id = %d #parents = %d chikdId = %d d = %f\n", i, DustParentChildCPU[i].numberOfParents, DustParentChildCPU[i].childId[0], d);
			}*/
		}
		// exit(0);
	}

	// Zach take these two to window
	if(key == 'l') // single dust selection
	{
		View = 1;
		topView();
		orthogonalView();
		MouseOn = 1;
		Pause = 1;
		SingleOrPairOfDust = 1;
	}
	
	if(key == 'L') // pair of dust selection, turned off in this demo
	{
		View = 1;
		topView();
		orthogonalView();
		MouseOn = 1;
		Pause = 1;
		SingleOrPairOfDust = 2;
	}

	if (key == '6')
	{
		if (DrawRate > 1) // Stops us from setting draw rate lower than 1.
			DrawRate--;
	}
	if (key == '7')
	{
		DrawRate++; // Guess there doesn't have to be a cap on this.
	}
	
	// Zack take all 3 to window and set the error if ffmpeg is not installed
	// I think we should figure out a way to perform some kind of error checking here if possible.
	// According to what I'm reading, unless opening a shell causes some kind of error or the shell just can't
	// call the command, it won't return null, so it won't crash trying to do anything unless it runs and fails.
	// So, if you don't have ffmpeg installed, and you press m, there's nothing to check that ffmpeg is installed or
	// can start without fail. Instead, it's just gonna fail and crash the program when you're trying to record the perfect
	// torsion.
	if(key == 'm')
	{
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
		MovieOn = 1;
	}
	if(key == 'M')
	{
		if(MovieOn == 1) 
		{
			pclose(MovieFile);
			printf("\nMovie was successfully completed.\n");
		}
		free(Buffer);
		
		MovieOn = 0;
	}
	if(key == 's')
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
	
	//Adjusting point change
	if(key == 'i')
	{
		BaseIonWakeFractionalCharge -= 0.01;
		if(BaseIonWakeFractionalCharge < 0.0) BaseIonWakeFractionalCharge = 0.0;
		//for(int i = 0; i < NumberOfDustParticles; i++)
		//{
			//IonWakeCPU[i].w = BaseIonWakeFractionalCharge;
		//}
		//printf("\n BaseIonWakeFractionalCharge = %f \n", BaseIonWakeFractionalCharge);
		//cudaMemcpy( IonWakeGPU, IonWakeCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
    	//errorCheck("cudaMemcpy IonWakeGPU up");
    	terminalPrint();
		// drawPicture();
	}
	if(key == 'I')
	{
		BaseIonWakeFractionalCharge += 0.01;
		//for(int i = 0; i < NumberOfDustParticles; i++)
		//{
			//IonWakeCPU[i].w = BaseIonWakeFractionalCharge;
		//}
		//printf("\n BaseIonWakeFractionalCharge = %f \n", BaseIonWakeFractionalCharge);
		//cudaMemcpy( IonWakeGPU, IonWakeCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
    	//errorCheck("cudaMemcpy IonWakeGPU up");
    	terminalPrint();
		// drawPicture();
	}
	
	// Adjusting pressure
	double deltaPressure = 1;
	if (key == 'P') // increase pressure
	{
		GasPressure += deltaPressure;
		//printf("\n Gas Pressure in millitorr = %f", GasPressure);
		Drag = PressureConstant * GasPressure;
		terminalPrint();
	}
	if (key == 'p') // decrease pressure
	{
		GasPressure -= deltaPressure;
		if (GasPressure < 0.0) GasPressure = 0.0;
		//printf("\n Gas Pressure in millitorr = %f", GasPressure);
		Drag = PressureConstant * GasPressure;
		terminalPrint();
	}
	
	// Adjusting power on lower plate
	double deltaBottomPlateForcingParameter = 1.0e7;
	if(key == 'V')
	{
		deltaBottomPlateForcingParameter = deltaBottomPlateForcingParameter*TimeUnit*TimeUnit*ChargeUnit/MassUnit;
		BottomPlateForcingParameter += deltaBottomPlateForcingParameter;
		//printf("\nBottomPlateForcingParameter = %e", BottomPlateForcingParameter/(TimeUnit*TimeUnit*ChargeUnit/MassUnit));
		terminalPrint();
	}
	if(key == 'v')
	{
		deltaBottomPlateForcingParameter = deltaBottomPlateForcingParameter*TimeUnit*TimeUnit*ChargeUnit/MassUnit;
		BottomPlateForcingParameter -= deltaBottomPlateForcingParameter;
		if(BottomPlateForcingParameter < 0.0) BottomPlateForcingParameter = 0.0;
		//printf("\nBottomPlateForcingParameter = %e", BottomPlateForcingParameter/(TimeUnit*TimeUnit*ChargeUnit/MassUnit));
		terminalPrint();
	}
	
	// Adjusting power on side of cavity
	double deltaRadialConfinementStrength = 1.0e5;
	if(key == 'C')
	{
		deltaRadialConfinementStrength = deltaRadialConfinementStrength*TimeUnit*TimeUnit*ChargeUnit/MassUnit;
		RadialConfinementStrength += deltaRadialConfinementStrength;
		//printf("\RadialConfinementStrength = %e", RadialConfinementStrength/(TimeUnit*TimeUnit*ChargeUnit/MassUnit));
		terminalPrint();
	}
	if(key == 'c')
	{
		deltaRadialConfinementStrength = deltaRadialConfinementStrength*TimeUnit*TimeUnit*ChargeUnit/MassUnit;
		RadialConfinementStrength -= deltaRadialConfinementStrength;
		if(RadialConfinementStrength < 0.0) RadialConfinementStrength = 0.0;
		//printf("\RadialConfinementStrength = %e", RadialConfinementStrength/(TimeUnit*TimeUnit*ChargeUnit/MassUnit));
		terminalPrint();
	}

	if (key == '9')
	{
		// frustrum
		frustrumView();
	}
	if (key == '0')
	{
		// Orthogonal
		orthogonalView();
	}

	if (SelectedDustMove) // This will definitely always be checked.
	{
		// Copy down dust grain positions.
		cudaMemcpy( DustPositionCPU, DustPositionGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
		errorCheck("cudaMemcpy DustPositionCPU down");
		cudaMemcpy( DustVelocityCPU, DustVelocityGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
		errorCheck("cudaMemcpy DustVelocityCPU down");
		cudaMemcpy( DustForceCPU, DustForceGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
		errorCheck("cudaMemcpy DustForceCPU down");
		cudaMemcpy( IonWakeCPU, IonWakeGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
		errorCheck("cudaMemcpy IonWakeCPU down");

		if(key == 'x')
		{
			DustPositionCPU[SelectedDustGrainId1].x += dx;
			DustForceCPU[SelectedDustGrainId1].x -= dx;
			//printf("Updated position: %f, %f, %f\n", 
											//DustPositionCPU[SelectedDustGrainId1].x, DustPositionCPU[SelectedDustGrainId1].y,
											//DustPositionCPU[SelectedDustGrainId1].z);
			terminalPrint();
			drawPicture();
		}
		if(key == 'X')
		{
			DustPositionCPU[SelectedDustGrainId1].x += -dx;
			//printf("Updated position: %f, %f, %f\n", 
											//DustPositionCPU[SelectedDustGrainId1].x, DustPositionCPU[SelectedDustGrainId1].y,
											//DustPositionCPU[SelectedDustGrainId1].z);
			terminalPrint();
			drawPicture();
		}
		
		if(key == 'y')
		{
		
			DustPositionCPU[SelectedDustGrainId1].y += dy;
			//printf("Updated position: %f, %f, %f\n", 
											//DustPositionCPU[SelectedDustGrainId1].x, DustPositionCPU[SelectedDustGrainId1].y,
											//DustPositionCPU[SelectedDustGrainId1].z);
			terminalPrint();
			drawPicture();
		}
		if(key == 'Y')
		{
			DustPositionCPU[SelectedDustGrainId1].y += -dy;
			//printf("Updated position: %f, %f, %f\n", 
											//DustPositionCPU[SelectedDustGrainId1].x, DustPositionCPU[SelectedDustGrainId1].y,
											//DustPositionCPU[SelectedDustGrainId1].z);
			terminalPrint();
			drawPicture();
		}
		
		if(key == 'z')
		{
			DustPositionCPU[SelectedDustGrainId1].z += dz;
			//printf("Updated position: %f, %f, %f\n", 
										//	DustPositionCPU[SelectedDustGrainId1].x, DustPositionCPU[SelectedDustGrainId1].y,
											//DustPositionCPU[SelectedDustGrainId1].z);
			terminalPrint();
			drawPicture();
		}
		if(key == 'Z')
		{
			DustPositionCPU[SelectedDustGrainId1].z += -dz;
			//printf("Updated position: %f, %f, %f\n", 
											//DustPositionCPU[SelectedDustGrainId1].x, DustPositionCPU[SelectedDustGrainId1].y,
											//DustPositionCPU[SelectedDustGrainId1].z);
			terminalPrint();
			drawPicture();
		}

		// In case we did modify the position, we push back up to GPU.
		cudaMemcpy( DustPositionGPU, DustPositionCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
		errorCheck("cudaMemcpy DustPositionCPU up");
		cudaMemcpy( DustVelocityGPU, DustVelocityCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
		errorCheck("cudaMemcpy DustVelocityGPU up");
		cudaMemcpy( DustForceGPU, DustForceCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
		errorCheck("cudaMemcpy DustForceGPU up");
		cudaMemcpy( IonWakeGPU, IonWakeCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
		errorCheck("cudaMemcpy IonWakeGPU up");
	}
	else 
	{
		dx *= 4.0f;
		dy *= 4.0f;
		dz *= 4.0f;
		if(key == 'x')
		{

			glTranslatef(dx, 0.0, 0.0);
			CenterOfView.x += dx;
			drawPicture();
		}
		if(key == 'X')
		{
			glTranslatef(-dx, 0.0, 0.0);
			CenterOfView.x += -dx;
			drawPicture();
		}
		
		if(key == 'y')
		{
			glTranslatef(0.0, dy, 0.0);
			CenterOfView.x += dy;
			drawPicture();
		}
		if(key == 'Y')
		{
			glTranslatef(0.0, -dy, 0.0);
			CenterOfView.x += -dy;
			drawPicture();
		}
		
		if(key == 'z')
		{
			glTranslatef(0.0, 0.0, dz);
			CenterOfView.x += dz;
			drawPicture();
		}
		if(key == 'Z')
		{
			glTranslatef(0.0, 0.0, -dz);
			CenterOfView.x += -dz;
			drawPicture();
		}
	}
	
	if(key == '1')
	{
		SelectedDustMove = 1;
		// printf("You turned on x/y/z movement for the selected dust grain.\n");
		/*cudaMemcpy( DustPositionCPU, DustPositionGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
		errorCheck("cudaMemcpy DustPositionCPU down");
		cudaMemcpy( DustVelocityCPU, DustVelocityGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
		errorCheck("cudaMemcpy DustVelocityCPU down");
		cudaMemcpy( DustForceCPU, DustForceGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
		errorCheck("cudaMemcpy DustForceCPU down");
		cudaMemcpy( IonWakeCPU, IonWakeGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
		errorCheck("cudaMemcpy IonWakeCPU down");
		
		charge = DustPositionCPU[SelectedDustGrainId1].w;
		diameter = DustVelocityCPU[SelectedDustGrainId1].w;
		numberOfElectrons = charge/ElectronCharge;
		electronsPerUnitDiameter = numberOfElectrons/diameter;
		
		diameter -= 0.0001;
		radius = diameter/2.0;
		mass = DustDensity*(4.0/3.0)*PI*radius*radius*radius;
		charge = ElectronCharge*round(electronsPerUnitDiameter*diameter);
		
		DustVelocityCPU[SelectedDustGrainId1].w = diameter; // Velocity w holds the diameter
		DustForceCPU[SelectedDustGrainId1].w = mass; // Force w holds the mass
		DustPositionCPU[SelectedDustGrainId1].w = charge; // Position holds the charge.
		
		printf("\n Diameter 0f dust grain %d = %f", SelectedDustGrainId1, (DustVelocityCPU[SelectedDustGrainId1].w*LengthUnit)*1.0e6);
		
		// Copying every thing back up to the GPU.
		cudaMemcpy( DustPositionGPU, DustPositionCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
		errorCheck("cudaMemcpy DustPositionCPU up");
		cudaMemcpy( DustVelocityGPU, DustVelocityCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
		errorCheck("cudaMemcpy DustVelocityGPU up");
		cudaMemcpy( DustForceGPU, DustForceCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
		errorCheck("cudaMemcpy DustForceGPU up");
		cudaMemcpy( IonWakeGPU, IonWakeCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
		errorCheck("cudaMemcpy IonWakeGPU up");*/
		terminalPrint();
		drawPicture();
	}
	if(key == '!')
	{
		SelectedDustMove = 0;
		printf("You turned off x/y/z movement for the selected dust grain.\n");
		/*cudaMemcpy( DustPositionCPU, DustPositionGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
		errorCheck("cudaMemcpy DustPositionCPU down");
		cudaMemcpy( DustVelocityCPU, DustVelocityGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
		errorCheck("cudaMemcpy DustVelocityCPU down");
		cudaMemcpy( DustForceCPU, DustForceGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
		errorCheck("cudaMemcpy DustForceCPU down");
		cudaMemcpy( IonWakeCPU, IonWakeGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
		errorCheck("cudaMemcpy IonWakeCPU down");
		
		charge = DustPositionCPU[SelectedDustGrainId1].w;
		diameter = DustVelocityCPU[SelectedDustGrainId1].w;
		numberOfElectrons = charge/ElectronCharge;
		electronsPerUnitDiameter = numberOfElectrons/diameter;
		
		diameter += 0.0001;
		radius = diameter/2.0;
		mass = DustDensity*(4.0/3.0)*PI*radius*radius*radius;
		charge = ElectronCharge*round(electronsPerUnitDiameter*diameter);
		
		DustVelocityCPU[SelectedDustGrainId1].w = diameter; // Velocity w holds the diameter
		DustForceCPU[SelectedDustGrainId1].w = mass; // Force w holds the mass
		DustPositionCPU[SelectedDustGrainId1].w = charge; // Position holds the charge.
		
		printf("\n Diameter 0f dust grain %d = %f", SelectedDustGrainId1, (DustVelocityCPU[SelectedDustGrainId1].w*LengthUnit)*1.0e6);
		*/

		// Copying every thing back up to the GPU.
		cudaMemcpy( DustPositionGPU, DustPositionCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
		errorCheck("cudaMemcpy DustPositionCPU up");
		cudaMemcpy( DustVelocityGPU, DustVelocityCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
		errorCheck("cudaMemcpy DustVelocityGPU up");
		cudaMemcpy( DustForceGPU, DustForceCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
		errorCheck("cudaMemcpy DustForceGPU up");
		cudaMemcpy( IonWakeGPU, IonWakeCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
		errorCheck("cudaMemcpy IonWakeGPU up");
		terminalPrint();
		drawPicture();
	}
	
	 // Off for this demo
 	
 	double diameter, radius, mass, charge, electronsPerUnitDiameter, numberOfElectrons;
	if(key == '2')
	{
		cudaMemcpy( DustPositionCPU, DustPositionGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
		errorCheck("cudaMemcpy DustPositionCPU down");
		cudaMemcpy( DustVelocityCPU, DustVelocityGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
		errorCheck("cudaMemcpy DustVelocityCPU down");
		cudaMemcpy( DustForceCPU, DustForceGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
		errorCheck("cudaMemcpy DustForceCPU down");
		cudaMemcpy( IonWakeCPU, IonWakeGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
		errorCheck("cudaMemcpy IonWakeCPU down");
		
		charge = DustPositionCPU[SelectedDustGrainId2].w;
		diameter = DustVelocityCPU[SelectedDustGrainId2].w;
		numberOfElectrons = charge/ElectronCharge;
		electronsPerUnitDiameter = numberOfElectrons/diameter;
		
		diameter -= 0.0001;
		radius = diameter/2.0;
		mass = DustDensity*(4.0/3.0)*PI*radius*radius*radius;
		charge = ElectronCharge*round(electronsPerUnitDiameter*diameter);
		
		DustVelocityCPU[SelectedDustGrainId2].w = diameter; // Velocity w holds the diameter
		DustForceCPU[SelectedDustGrainId2].w = mass; // Force w holds the mass
		DustPositionCPU[SelectedDustGrainId2].w = charge; // Position holds the charge.
		
		printf("\n Diameter 0f dust grain %d = %f", SelectedDustGrainId2, (DustVelocityCPU[SelectedDustGrainId2].w*LengthUnit)*1.0e6);
		
		charge = DustPositionCPU[SelectedDustGrainId3].w;
		diameter = DustVelocityCPU[SelectedDustGrainId3].w;
		numberOfElectrons = charge/ElectronCharge;
		electronsPerUnitDiameter = numberOfElectrons/diameter;
		
		diameter += 0.0001;
		radius = diameter/2.0;
		mass = DustDensity*(4.0/3.0)*PI*radius*radius*radius;
		charge = ElectronCharge*round(electronsPerUnitDiameter*diameter);
		
		DustVelocityCPU[SelectedDustGrainId3].w = diameter; // Velocity w holds the diameter
		DustForceCPU[SelectedDustGrainId3].w = mass; // Force w holds the mass
		DustPositionCPU[SelectedDustGrainId2].w = charge; // Position holds the charge.
		
		printf("\n Diameter 0f dust grain %d = %f", SelectedDustGrainId3, (DustVelocityCPU[SelectedDustGrainId3].w*LengthUnit)*1.0e6);
		
		// Copying every thing back up to the GPU.
		cudaMemcpy( DustPositionGPU, DustPositionCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
		errorCheck("cudaMemcpy DustPositionCPU up");
		cudaMemcpy( DustVelocityGPU, DustVelocityCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
		errorCheck("cudaMemcpy DustVelocityGPU up");
		cudaMemcpy( DustForceGPU, DustForceCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
		errorCheck("cudaMemcpy DustForceGPU up");
		cudaMemcpy( IonWakeGPU, IonWakeCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
		errorCheck("cudaMemcpy IonWakeGPU up");
		
		drawPicture();
	}
	if(key == '@')
	{
		cudaMemcpy( DustPositionCPU, DustPositionGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
		errorCheck("cudaMemcpy DustPositionCPU down");
		cudaMemcpy( DustVelocityCPU, DustVelocityGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
		errorCheck("cudaMemcpy DustVelocityCPU down");
		cudaMemcpy( DustForceCPU, DustForceGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
		errorCheck("cudaMemcpy DustForceCPU down");
		cudaMemcpy( IonWakeCPU, IonWakeGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
		errorCheck("cudaMemcpy IonWakeCPU down");
		
		charge = DustPositionCPU[SelectedDustGrainId2].w;
		diameter = DustVelocityCPU[SelectedDustGrainId2].w;
		numberOfElectrons = charge/ElectronCharge;
		electronsPerUnitDiameter = numberOfElectrons/diameter;
		
		diameter += 0.0001;
		radius = diameter/2.0;
		mass = DustDensity*(4.0/3.0)*PI*radius*radius*radius;
		charge = ElectronCharge*round(electronsPerUnitDiameter*diameter);
		
		DustVelocityCPU[SelectedDustGrainId2].w = diameter; // Velocity w holds the diameter
		DustForceCPU[SelectedDustGrainId2].w = mass; // Force w holds the mass
		DustPositionCPU[SelectedDustGrainId2].w = charge; // Position holds the charge.
		
		printf("\n Diameter 0f dust grain %d = %f", SelectedDustGrainId2, (DustVelocityCPU[SelectedDustGrainId2].w*LengthUnit)*1.0e6);
		
		charge = DustPositionCPU[SelectedDustGrainId3].w;
		diameter = DustVelocityCPU[SelectedDustGrainId3].w;
		numberOfElectrons = charge/ElectronCharge;
		electronsPerUnitDiameter = numberOfElectrons/diameter;
		
		diameter -= 0.0001;
		radius = diameter/2.0;
		mass = DustDensity*(4.0/3.0)*PI*radius*radius*radius;
		charge = ElectronCharge*round(electronsPerUnitDiameter*diameter);
		
		DustVelocityCPU[SelectedDustGrainId3].w = diameter; // Velocity w holds the diameter
		DustForceCPU[SelectedDustGrainId3].w = mass; // Force w holds the mass
		DustPositionCPU[SelectedDustGrainId2].w = charge; // Position holds the charge.
		
		printf("\n Diameter 0f dust grain %d = %f", SelectedDustGrainId3, (DustVelocityCPU[SelectedDustGrainId3].w*LengthUnit)*1.0e6);
		
		// Copying every thing back up to the GPU.
		cudaMemcpy( DustPositionGPU, DustPositionCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
		errorCheck("cudaMemcpy DustPositionCPU up");
		cudaMemcpy( DustVelocityGPU, DustVelocityCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
		errorCheck("cudaMemcpy DustVelocityGPU up");
		cudaMemcpy( DustForceGPU, DustForceCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
		errorCheck("cudaMemcpy DustForceGPU up");
		cudaMemcpy( IonWakeGPU, IonWakeCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
		errorCheck("cudaMemcpy IonWakeGPU up");
		
		drawPicture();
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
	float dustX, dustY, dustZ;
	float closest;
	float dx, dy, dz, d;
	double diameter, radius, mass, charge, electronsPerUnitDiameter;
	
	if(state == GLUT_DOWN)
	{
		if(MouseOn == 1)
		{
			// You need to copy everything down from the GPU because the mass is in force.w the diameter is in vel.w and the charge is in pos.w
			cudaMemcpy( DustPositionCPU, DustPositionGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
			errorCheck("cudaMemcpy DustPositionCPU down");
			cudaMemcpy( DustVelocityCPU, DustVelocityGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
			errorCheck("cudaMemcpy DustVelocityCPU down");
			cudaMemcpy( DustForceCPU, DustForceGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
			errorCheck("cudaMemcpy DustForceCPU down");
			cudaMemcpy( IonWakeCPU, IonWakeGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
			errorCheck("cudaMemcpy IonWakeCPU down");
			cudaMemcpy( DustParentChildCPU, DustParentChildGPU, NumberOfDustParticles*sizeof(dustParentChildStructure), cudaMemcpyDeviceToHost);
			errorCheck("cudaMemcpy DustParentChildCPU down");
			
			if(button == GLUT_LEFT_BUTTON)
			{
				if(SingleOrPairOfDust == 1) // Just selecting 1 Dust
				{
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
					
					// Finding the closest dust grain to where you mouse clicked.
					closest = 10000000.0;

					// ---------------------------
					if (SelectedDustGrainId1 != -1) // Color our old dust a pinkish color.
					{
						PreviouslySelectedGrain = SelectedDustGrainId1;
						DustColor[PreviouslySelectedGrain].x = 0.941f;
						DustColor[PreviouslySelectedGrain].y = 0.000f;
						DustColor[PreviouslySelectedGrain].z = 1.000f;
					}
					// ---------------------------

					SelectedDustGrainId1 = -1;
					for(int i = 1; i < NumberOfDustParticles; i++)
					{
						dx = DustPositionCPU[i].x - myX;
						dz = DustPositionCPU[i].z - myZ;
						d = sqrt(dx*dx + dz*dz);
						if(d < closest)
						{
							closest = d;
							SelectedDustGrainId1 = i;
						}
					}
					
					// Setting the color of the selected dust and printing out it's location.
					if(SelectedDustGrainId1 != -1)
					{
						// Coloring the dust grain blue so you know which one you changed.
						DustColor[SelectedDustGrainId1].x = 0.0;
						DustColor[SelectedDustGrainId1].y = 0.0;
						DustColor[SelectedDustGrainId1].z = 1.0;
						dustX = DustPositionCPU[SelectedDustGrainId1].x*LengthUnit;
						dustY = DustPositionCPU[SelectedDustGrainId1].y*LengthUnit;
						dustZ = DustPositionCPU[SelectedDustGrainId1].z*LengthUnit;
						printf("\n\n DustGrain: x = %f, y = %f, z = %f", dustX, dustY, dustZ);
						/*for (int i = 0; i < 10; i++)
						{
							int child = DustParentChildCPU[SelectedDustGrainId1].childId[i];
							if (child != -1)
							{
								DustColor[child].x = 1.0;
								DustColor[child].y = 0.0;
								DustColor[child].z = 1.0;
							}
							
						}*/
					}
					else
					{
						printf("\n Error no dust grain selected");
						exit(0);
					}
					drawPicture();
				}
				else if(SingleOrPairOfDust == 2)
				{
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
					
					// Finding the closest dust grain to where you mouse clicked.
					closest = 10000000.0;
					SelectedDustGrainId2 = -1;
					for(int i = 1; i < NumberOfDustParticles; i++)
					{
						dx = DustPositionCPU[i].x - myX;
						dz = DustPositionCPU[i].z - myZ;
						d = sqrt(dx*dx + dz*dz);
						if(d < closest)
						{
							closest = d;
							SelectedDustGrainId2 = i;
						}
					}
					
					// Setting the color of the selected dust and printing out it's location.
					if(SelectedDustGrainId2 != -1)
					{
						// Coloring the dust grain blue so you know which one you changed.
						DustColor[SelectedDustGrainId2].x = 0.0;
						DustColor[SelectedDustGrainId2].y = 1.0;
						DustColor[SelectedDustGrainId2].z = 0.0;
						dustX = DustPositionCPU[SelectedDustGrainId2].x*LengthUnit;
						dustY = DustPositionCPU[SelectedDustGrainId2].y*LengthUnit;
						dustZ = DustPositionCPU[SelectedDustGrainId2].z*LengthUnit;
						printf("\n\n Dust grain1: x = %f, y = %f, z = %f", dustX, dustY, dustZ);
					}
					else
					{
						printf("\n Error no dust grain selected");
						exit(0);
					}
				
					// Finding the closest dust grain to the dust grain just selected.
					closest = 10000000.0;
					SelectedDustGrainId3 = -1;
					for(int i = 1; i < NumberOfDustParticles; i++)
					{
						if(i != SelectedDustGrainId2)
						{
							dx = DustPositionCPU[i].x - DustPositionCPU[SelectedDustGrainId2].x;
							dy = DustPositionCPU[i].y - DustPositionCPU[SelectedDustGrainId2].y;
							dz = DustPositionCPU[i].z - DustPositionCPU[SelectedDustGrainId2].z;
							d = sqrt(dx*dx + dy*dy + dz*dz);
							if(d < closest)
							{
								closest = d;
								SelectedDustGrainId3 = i;
							}
						}
					}
					
					// Setting the color of the selected dust and printing out it's location.
					if(SelectedDustGrainId3 != -1)
					{
						// Coloring the dust grain blue so you know which one you changed.
						DustColor[SelectedDustGrainId3].x = 1.0;
						DustColor[SelectedDustGrainId3].y = 0.0;
						DustColor[SelectedDustGrainId3].z = 1.0;
						dustX = DustPositionCPU[SelectedDustGrainId3].x*LengthUnit;
						dustY = DustPositionCPU[SelectedDustGrainId3].y*LengthUnit;
						dustZ = DustPositionCPU[SelectedDustGrainId3].z*LengthUnit;
						printf("\n\n Dust grain2: x = %f, y = %f, z = %f", dustX, dustY, dustZ);
					}
					else
					{
						printf("\n Error no dust grain selected");
						exit(0);
					}
				
					// Setting both selected dust grains to the base diameter, mass, and charge.
					// This will start them at the base for a referance as they change.
					diameter = BaseDustDiameter;
					electronsPerUnitDiameter = BaseElectronsPerUnitDiameter;
					radius = diameter/2.0;
					mass = DustDensity*(4.0/3.0)*PI*radius*radius*radius;
					charge = ElectronCharge*round(electronsPerUnitDiameter*diameter);
					
					DustVelocityCPU[SelectedDustGrainId2].w = diameter; // Velocity w holds the diameter
					DustForceCPU[SelectedDustGrainId2].w = mass; // Force w holds the mass
					DustPositionCPU[SelectedDustGrainId2].w = charge; // Position holds the charge.
					
					DustVelocityCPU[SelectedDustGrainId3].w = diameter; // Velocity w holds the diameter
					DustForceCPU[SelectedDustGrainId3].w = mass; // Force w holds the mass
					DustPositionCPU[SelectedDustGrainId3].w = charge; // Position holds the charge.
				}
				else
				{
					printf("\n Error bad SingleOrPairOfDust slection");
					exit(0);
				}
				
				drawPicture();
				MouseOn = 0;
				// Pause = 0;
				
				frustrumView();
				terminalPrint();
				
				// Copying every thing back up to the GPU.
				cudaMemcpy( DustPositionGPU, DustPositionCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
				errorCheck("cudaMemcpy DustPositionCPU up");
				cudaMemcpy( DustVelocityGPU, DustVelocityCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
				errorCheck("cudaMemcpy DustVelocityGPU up");
				cudaMemcpy( DustForceGPU, DustForceCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
				errorCheck("cudaMemcpy DustForceGPU up");
				cudaMemcpy( IonWakeGPU, IonWakeCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
				errorCheck("cudaMemcpy IonWakeGPU up");
				cudaMemcpy( DustParentChildGPU, DustParentChildCPU, NumberOfDustParticles*sizeof(dustParentChildStructure), cudaMemcpyHostToDevice);
				errorCheck("cudaMemcpy DustParentChildGPU up");
			}
		}
	}
}

