// nvcc horizontalDustDisk.cu -o dust -lglut -lm -lGLU -lGL
//To force kill hit "control c" in the window you launched it from.

#include <iostream>
#include <fstream>
#include <sstream>
#include <string.h>
#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <curand.h>
#include <curand_kernel.h>
#include <random>
using namespace std;

#define PI 3.141592654
#define BLOCK 256
#define ONE_GRAIN 0
#define TWO_GRAINS 1
#define START_MOVIE 2
#define STOP_MOVIE 3
#define TAKE_SCREENSHOT 4
#define HELP 5
#define MAX_NUMBER_CHILDREN 10
#define MAX_FRACTIONAL_ION_WAKE_CHARGE_GIVE 0.7f
#define EPSILON 0.000001f;
#define EPSILON2 1.0e-12f;

struct dustParentChildStructure
{
	int numberOfParents;
	int childId[MAX_NUMBER_CHILDREN];   
};

FILE* MovieFile;
int* Buffer;

// Globals to be read in from parameter file.
int	NumberOfDustParticles;

double 	Gravity;
double 	DustDensity;
double 	BaseDustDiameter;
double	DustDiameterStandardDeviation;
int 	DustDistributionType;

double 	CoulombConstant;
double	ElectronCharge;
double 	BaseElectronsPerUnitDiameter;
double	electronStdPerUnitDiameter;
double	IonDebyeLength;
double	CutOffMultiplier;
double	SheathHeight;

double 	BaseIonWakeFractionalCharge;
double 	BaseIonWakeLength;
double  ChargeExchangeFraction;

double 	RadialConfinementStrength;
double 	RadialConfinementScale;
double 	RadialConfinementHeight;
double 	BottomPlateForcingParameter;

double  BoltzmannConstant;
double  EpsteinConstant;
double  GasTemperature;
double  GasPressure;
double  NeutralGasMass;

double 	Dt;

int 	DrawRate;
int 	PrintTimeRate;

// Drag will be calculated from above values.
double 	Drag; 
double  PressureConstant;

// Globals to hold our unit convertions.
double MassUnit;
double LengthUnit;
double ChargeUnit;
double TimeUnit;

// Call back globals
int Pause;
int Trace;
int MouseOn;
int MovieOn;
int View; // 0 top view, 1 side view.
int RadialConfinementViewingAids;
float4 CenterOfView;
float4 AngleOfView;
int DrawBottomRing;
int SelectedDustGrainId1, SelectedDustGrainId2, SelectedDustGrainId3;
int PreviouslySelectedGrain; // Holds the last selected grain
int SingleOrPairOfDust;
int Freeze; // 1 means we are frozen, 0 means not frozen (so moveDust is called)
int SelectedDustMove; // 1 means xyz movement is on for selected dust, 0 means it is off.
// 1 means means a red/green line is drawn between parent dust and its companion (child). 
// and a blue line is draw from a dust to its ion point charge. 0 means no lines are drawn.
int DustViewingAids;

// Timing globals
int DrawTimer;
int PrintTimer;
float RunTime;

// Position, velocity, force and ionwake globals
float4 *DustPositionCPU, *DustVelocityCPU, *DustForceCPU, *IonWakeCPU;
float4 *DustPositionGPU, *DustVelocityGPU, *DustForceGPU, *IonWakeGPU;
float4 *DustColor;
// Holds information about the closest dust below the dust in question. Used to adjust their point charges.
dustParentChildStructure *DustParentChildCPU;
dustParentChildStructure *DustParentChildGPU;
int *DustUsedCountCPU;
int *DustUsedCountGPU;

// CUDA globals
dim3 Block, Grid;

// Window globals
static int Window;
int XWindowSize;
int YWindowSize; 
double Near;
double Far;
double EyeX;
double EyeY;
double EyeZ;
double CenterX;
double CenterY;
double CenterZ;
double UpX;
double UpY;
double UpZ;

// Prototyping functions
void readSimulationParameters();
void setUnitConvertions();
void PutConstantsIntoOurUnits();
void allocateMemory();
void setInitialConditions(); 
void drawPicture();
void n_body();
void terminalPrint();
void errorCheck(const char*);
void setup();
void topView();
void sideView();

#include "./callBackFunctions.h"

// Function that reads in the variable from the setuo file.
void readSimulationParameters()
{
	ifstream data;
	string name;
	
	data.open("./simulationSetup");
	
	if(data.is_open() == 1)
	{
		getline(data,name,'=');
		data >> NumberOfDustParticles;
		
		getline(data,name,'=');
		data >> Gravity;
		
		getline(data,name,'=');
		data >> DustDensity;
		
		getline(data,name,'=');
		data >> BaseDustDiameter;
		
		getline(data,name,'=');
		data >> DustDiameterStandardDeviation;

		getline(data, name, '=');
		data >> DustDistributionType;
		
		getline(data,name,'=');
		data >> CoulombConstant;
		
		getline(data,name,'=');
		data >> ElectronCharge;
		
		getline(data,name,'=');
		data >> BaseElectronsPerUnitDiameter;
		
		getline(data,name,'=');
		data >> electronStdPerUnitDiameter;
		
		getline(data,name,'=');
		data >> IonDebyeLength;
		
		getline(data,name,'=');
		data >> CutOffMultiplier;
		
		getline(data,name,'=');
		data >> SheathHeight;
		
		getline(data,name,'=');
		data >> BaseIonWakeFractionalCharge;
		
		getline(data,name,'=');
		data >> ChargeExchangeFraction;
		
		getline(data,name,'=');
		data >> BaseIonWakeLength;
		
		getline(data,name,'=');
		data >> RadialConfinementStrength;
		
		getline(data,name,'=');
		data >> RadialConfinementScale;
		
		getline(data,name,'=');
		data >> RadialConfinementHeight;
		
		getline(data,name,'=');
		data >> BottomPlateForcingParameter;
		
		getline(data,name,'=');
		data >> BoltzmannConstant;  
		
		getline(data,name,'=');
		data >> EpsteinConstant;
		
		getline(data,name,'=');
		data >> GasTemperature;
		
		getline(data,name,'=');
		data >> GasPressure;
		
		getline(data,name,'=');
		data >> NeutralGasMass;
		
		getline(data,name,'=');
		data >> Dt;
		
		getline(data,name,'=');
		data >> DrawRate;
		
		getline(data,name,'=');
		data >> PrintTimeRate;
	}
	else
	{
		printf("\nTSU Error could not open simulationSetup file\n");
		exit(0);
	}
	
	// Printing out all the values read in for a spot check of the program.
	printf("\n These are what all the basic constants that were read in for a spot check\n");
	printf("\n NumberOfDustParticles.......................... %d", NumberOfDustParticles);
	printf("\n Gravity........................................ %f meters/second²", Gravity);
	printf("\n DustDensity.................................... %f grams/centimeter³", DustDensity);
	printf("\n BaseDustDiameter............................... %f microns", BaseDustDiameter);
	printf("\n DustDiameterStandardDeviation.................. %f microns", DustDiameterStandardDeviation);
	printf("\n DustDistributionType........................... %d ", DustDistributionType);
	printf("\n CoulombConstant................................ %e grams*meters³*second^-2*coulomb^-2", CoulombConstant);
	printf("\n ElectronCharge................................. %e coulombs", ElectronCharge);
	printf("\n BaseElectronsPerUnitDiameter................... %f electrons per micron", BaseElectronsPerUnitDiameter);
	printf("\n BaseElectronsPerUnitDiameter*BaseDustDiameter.. %f number of electrons on a standard dust grain.", BaseElectronsPerUnitDiameter*BaseDustDiameter);
	printf("\n electronStdPerUnitDiameter..................... %f electron fluxuation per micron", electronStdPerUnitDiameter);
	printf("\n electronStdPerUnitDiameter*BaseDiameter........ %f electron fluxuation on a standard dust grain", electronStdPerUnitDiameter*BaseDustDiameter);
	printf("\n IonDebyeLength................................. %f microns", IonDebyeLength);
	printf("\n CutOffMultiplier............................... %f", CutOffMultiplier);
	printf("\n SheathHeight................................... %f microns", SheathHeight);
	printf("\n BaseIonWakeFractionalCharge.................... %f fraction of dust charge", BaseIonWakeFractionalCharge);
	printf("\n ChargeExchangeFraction......................... %f fraction of charge transfer", ChargeExchangeFraction);
	printf("\n BaseIonWakeLength.............................. %f microns", BaseIonWakeLength);
	printf("\n RadialConfinementStrength...................... %e grams·meters·secondE-2CoulombE-1", RadialConfinementStrength);
	printf("\n RadialConfinementScale......................... %f centimeters", RadialConfinementScale);
	printf("\n RadialConfinementHeight........................ %f centimeters", RadialConfinementHeight);
	printf("\n BottomPlateForcingParameter.................... %e grams*secondE-2*CoulombE-1", BottomPlateForcingParameter);
	printf("\n Drag........................................... %e ???", Drag);
	printf("\n Dt = %f number of divisions of the final time unit.", Dt);
	printf("\n DrawRate = %d Dts between picture draws", DrawRate);
	printf("\n PrintTimeRate = %d Dts between time prints to the screen", PrintTimeRate);
	
	data.close();
	printf("\n\n ********************************************************************************");
	printf("\n Parameter file has been read");
	printf("\n ********************************************************************************\n");
}

// Allocating the memory needed on the CPU and GPU to run the program.
void allocateMemory()
{
	Block.x = BLOCK;
	Block.y = 1;
	Block.z = 1;
	
	Grid.x = (NumberOfDustParticles - 1)/Block.x + 1;
	Grid.y = 1;
	Grid.z = 1;
	
	DustPositionCPU = (float4*)malloc(NumberOfDustParticles*sizeof(float4));
	DustVelocityCPU = (float4*)malloc(NumberOfDustParticles*sizeof(float4));
	DustForceCPU    = (float4*)malloc(NumberOfDustParticles*sizeof(float4));
	IonWakeCPU    	= (float4*)malloc(NumberOfDustParticles*sizeof(float4));
	DustColor    	= (float4*)malloc(NumberOfDustParticles*sizeof(float4));
	DustParentChildCPU = (dustParentChildStructure*)malloc(NumberOfDustParticles*sizeof(dustParentChildStructure));
	DustUsedCountCPU = (int*)malloc(1*sizeof(int));
	
	cudaMalloc( (void**)&DustPositionGPU, NumberOfDustParticles*sizeof(float4));
	errorCheck("cudaMalloc DustPositionGPU");
	cudaMalloc( (void**)&DustVelocityGPU, NumberOfDustParticles*sizeof(float4));
	errorCheck("cudaMalloc DustVelocityGPU");
	cudaMalloc( (void**)&DustForceGPU,    NumberOfDustParticles*sizeof(float4));
	errorCheck("cudaMalloc DustForceGPU");
	cudaMalloc( (void**)&IonWakeGPU,    NumberOfDustParticles*sizeof(float4));
	errorCheck("cudaMalloc IonWakeGPU");
	cudaMalloc( (void**)&DustParentChildGPU, NumberOfDustParticles*sizeof(dustParentChildStructure));
	errorCheck("cudaMalloc DustParentChildGPU");
	cudaMalloc( (void**)&DustUsedCountGPU, 1*sizeof(int));
	errorCheck("cudaMalloc DustUsedCountGPU");
	
	printf("\n\n ********************************************************************************");
	printf("\n Memory has been allocated");
	printf("\n ********************************************************************************\n");
}

// This function creates a normalized unit system.
void setUnitConvertions()
{	
	/*
	Mass:
	Because the only mass we will be dealing with is a dust particle we will let the mass of a dust particle be 1. 
	Now how many grams is that? 
	The density of a dust particle is DensityOfDustParticle g/cm^3. 
	The diameter of a dust particle is DiameterOfDustParticle microns.
	If we assume the shape of a dust particlecto be a sphere the volume will be 4/3*PI*r^3.
	Lets get micometers to cm so the radius of a dust particle is (DiameterOfDustParticle/2.0)E-4 cm.
	Hence the mass of a dust particle is DensityOfDustParticle*g*(cm^-3)*(4.0/3.0)*PI*((DiameterOfDustParticle/2.0)E-12)*(cm^3).
	So one of our mass units is DensityOfDustParticle*(4.0/3.0)*PI*((DiameterOfDustParticle/2.0)E-12) grams.
	*/
	MassUnit = (DustDensity*(4.0/3.0)*PI*(BaseDustDiameter/(2.0*10000.0))*(BaseDustDiameter/(2.0*10000.0))*(BaseDustDiameter/(2.0*10000.0)));
	
	/*
	Length:
	Because the most important distance is the distance between two adjacent dust particles at equalibrium we will let that be our distance unit.
	To get started let assume we are working with a RadialConfinementScale cm radius well and a NumberOfDustParticles dust particles.
	For simplicity lets put an evenly spaced square latice in a RadialConfinementScale circle. 
	The sides would be 2*RadialConfinementScale/(2^(1/2)) cm long.
	The number of dust particle in each row would be NumberOfDustParticles^(1/2).
	The number of spacing in each row would be NumberOfDustParticles^(1/2) - 1.
	Hense the length of each spacing would be RadialConfinementScale*(2^(1/2))/(NumberOfDustParticles^(1/2) - 1) centimeters.
	Lets put this in meters.
	So our length unit is RadialConfinementScale*(2^(1/2))/(NumberOfDustParticles^(1/2) - 1)/100 meters.
	*/
	LengthUnit = ((2.0*RadialConfinementScale/sqrt(2.0))/(sqrt(NumberOfDustParticles) - 1.0))/100.0;
	
	/*
	Charge:
	Because the most important charge is the charge on a dust particle this should be around our charge unit.
	If we assume n electrons per micro of dust particle diameter we have our charge unit as 
	DiameterOfDustParticle(in microns)*n*1.60217662E-19 coulombs.
	So our charge unit is n*DiameterOfDustParticle*1.602E-19 C.
	Note, this a positive number.
	*/
	ChargeUnit = BaseElectronsPerUnitDiameter*BaseDustDiameter*(-ElectronCharge);
	
	/*
	Time:
	Let's set time so that the Coulomb constant is 1. That just makes calulations easier.
	Assuming K at 8.9875517923E9 kg*m^3*s^-2*C^-2 or 8.9875517923E12 g*m^3*s^-2*C^-2*
	If I did every thing right I get our time unit should be the square root of massUnit*LengthUnit^3*ChargeUnit^-2*(8.9875517923E12)^-1 seconds.
	*/
	TimeUnit = sqrt(MassUnit*LengthUnit*LengthUnit*LengthUnit/(ChargeUnit*ChargeUnit*CoulombConstant));
	
	// To change a mass, length, charge or time from the run units into grams, meters, coulombs or seconds just multiple by the apropriate unit from here.
	// To change a mass, length, charge or time from grams, meters, coulombs or seconds into our units just divide by the apropriate unit from here.
	printf("\n This is what all of the convertions are for a spot check\n");
	printf("\n Our MassUnit 	= %e grams", MassUnit);
	printf("\n Our LengthUnit = %e meters", LengthUnit);
	printf("\n Our ChargeUnit = %e coulombs", ChargeUnit);
	printf("\n Our TimeUnit 	= %e seconds", TimeUnit);
	
	printf("\n\n ********************************************************************************");
	printf("\n Unit convertions have been set.");
	printf("\n ********************************************************************************\n");
}

// This function converts everything into the normalized units.
void PutConstantsIntoOurUnits()
{
	// NumberOfDustParticles is just a number so no need to convert.
	
	// Gravity is in meters per secind square so in of units we need to multiplu by TimeUnit^2*LengthUnit^-1.
	Gravity *= TimeUnit*TimeUnit/LengthUnit;
	
	// Dust density is in grams per cenimeter cubed so need to take this to meters then divive by (MassUnit/LengthUnit^2);
	DustDensity *= 100.0*100.0*100.0;
	DustDensity /= (MassUnit/(LengthUnit*LengthUnit*LengthUnit));
	
	// BaseDustDiameter is in microns so take it to meters then divide by lengthUnit.
	BaseDustDiameter /= 1.0e6;
	BaseDustDiameter /= LengthUnit;
	
	// Standard deviation is in microns so take it to meters then divide by lengthUnit.
	DustDiameterStandardDeviation /= 1.0e6;
	DustDiameterStandardDeviation /= LengthUnit;
	
	// The coulomb constant should be 1 because of the way we set the time unit (see TimeUnit above).
	CoulombConstant = 1.0; 
	
	// Putting the charge of an electron into our units.
	ElectronCharge /= ChargeUnit;
	
	// Putting the number of electrons per unit of diameter into our units. They were in electrons per micron so take them to meters then to our units.
	BaseElectronsPerUnitDiameter *= 1.0e6;
	BaseElectronsPerUnitDiameter *= LengthUnit;
	
	// Putting the number of electrons per unit of diameter into our units. They were in electrons per micron so take them to meters then to our units.
	electronStdPerUnitDiameter *= 1.0e6;
	electronStdPerUnitDiameter *= LengthUnit;
	
	// Puting the debye length into our units. It is in micros so take it to meters then to our units.
	IonDebyeLength /= 1.0e6;
	IonDebyeLength /= LengthUnit;
	
	// Puting the sheath height into our units. It is in millimeters so take it to meters then to our units.
	SheathHeight /= 1.0e3;
	SheathHeight /= LengthUnit;
	
	// BaseIonWakeFractionalCharge No change needed here because this is a fraction of the dust charge it is attached to.
	
	// This is in microns so take it to meters then to our units.
	BaseIonWakeLength /= 1.0e6;
	BaseIonWakeLength /= LengthUnit;
	
	// This is a force we multiple by the dust charge to push it away from the confinement wall. It is given in grams*meters*seconds^-2*coulomb^-1.
	// To get it into our unit you need to divide by the apropriate units.
	RadialConfinementStrength = RadialConfinementStrength*TimeUnit*TimeUnit*ChargeUnit/(MassUnit*LengthUnit);
	
	// Taking RadialConfinement dimitions in cm to our units. First take them to meters then to our units.
	RadialConfinementScale /= 100.0;
	RadialConfinementScale /= LengthUnit;
	RadialConfinementHeight /= 100.0;
	RadialConfinementHeight /= LengthUnit;
	
	// This is the force that you multiply times the charge and distance from the bottom plate to get a force.
	// It is given in grams*seconds^-2*coulomb^-1.
	// To get it into our unit you need to divide by the apropriate units.
	BottomPlateForcingParameter = BottomPlateForcingParameter*TimeUnit*TimeUnit*ChargeUnit/MassUnit;
	

	// TODO: Read this in as a variable from the setup file.
	// I do not think pressure is correct. Does EpsteinConstant need to be converted?
	// Is Boltsmann's constant converted correctly???
	// This does not look like the paper????????????
	
	PressureConstant = (4.0/3.0)*sqrt(8.0)*EpsteinConstant*(1.0/sqrt(BoltzmannConstant*1000))*sqrt(PI*NeutralGasMass/6.022e23)*(101325.0/760.0)*(1.0/sqrt(GasTemperature))*((LengthUnit*LengthUnit*TimeUnit)/MassUnit);
	
	// This was what we had done.
	Drag = PressureConstant * GasPressure;

	// CutOffMultiplier is just a multiplier so no adjustment is needed.
	// Dt is a fraction of the time unit so no need to change
	// DrawRate is just a number of step between drawing so no change needed.
	// PrintTimeRate is just a number of step between Printing the time to the screen so no change needed.
	
	// Printing out just for a spot check
	printf("\n These are what all the basic constants in our units for a spot check\n");
	printf("\n NumberOfDustParticles = %d", NumberOfDustParticles);
	printf("\n Gravity = %e", Gravity);
	printf("\n DustDensity = %e", DustDensity);
	printf("\n BaseDustDiameter = %e", BaseDustDiameter);
	printf("\n DustDiameterStandardDeviation = %e", DustDiameterStandardDeviation);
	if (DustDistributionType == 0) printf("\n Using log normal distribution.");
	else printf("\n Using truncated normal distribution.");
	printf("\n CoulombConstant = %e", CoulombConstant);
	printf("\n ElectronCharge = %e", ElectronCharge);
	printf("\n BaseElectronsPerUnitDiameter = %e", BaseElectronsPerUnitDiameter);
	printf("\n BaseElectronsPerUnitDiameter*BaseDustDiameter = %e", BaseElectronsPerUnitDiameter*BaseDustDiameter);
	printf("\n electronStdPerUnitDiameter = %e", electronStdPerUnitDiameter);
	printf("\n electronStdPerUnitDiameter*BaseDiameter = %e", electronStdPerUnitDiameter*BaseDustDiameter);
	printf("\n IonDebyeLength = %e", IonDebyeLength);
	printf("\n CutOffMultiplier = %e", CutOffMultiplier);
	printf("\n SheathHeight = %e", SheathHeight);
	printf("\n BaseIonWakeFractionalCharge = %e", BaseIonWakeFractionalCharge);
	printf("\n ChargeExchangeFraction = %f fraction of charge transfer", ChargeExchangeFraction);
	printf("\n BaseIonWakeLength = %e", BaseIonWakeLength);
	printf("\n RadialConfinementStrength = %e", RadialConfinementStrength);
	printf("\n RadialConfinementScale = %e", RadialConfinementScale);
	printf("\n RadialConfinementHeight = %e", RadialConfinementHeight);
	printf("\n BottomPlateForcingParameter = %e", BottomPlateForcingParameter);
	printf("\n Drag = %e", Drag);
	printf("\n Dt = %e", Dt);
	printf("\n DrawRate = %d", DrawRate);
	printf("\n PrintTimeRate = %d", PrintTimeRate);
	
	printf("\n\n ********************************************************************************");
	printf("\n Constants have been put into our units.");
	printf("\n ********************************************************************************\n");
}

void setInitialConditions()
{
	int test;
	double temp1, temp2;
	double mag, radius, seperation;
	double diameter, mass, charge, randomNumber; // numberOfElectronsPerUnitDiameter;
	int numberOfElectrons;
	time_t t;
	
	// Seading the random number generater.
	srand((unsigned) time(&t));
	
	// Zeroing everything out just to be safe and setting the base color. 
	for(int i = 0; i < NumberOfDustParticles; i++)
	{
		// Alternatively:
		// memset(DustPositionCPU, 0, NumberOfDustParticles*sizeof(DustPositionCPU));
		DustPositionCPU[i].x = 0.0;
		DustPositionCPU[i].y = 0.0;
		DustPositionCPU[i].z = 0.0;
		DustPositionCPU[i].w = 0.0; // The w component of position holds the charge of the dust particle.
		
		DustVelocityCPU[i].x = 0.0;
		DustVelocityCPU[i].y = 0.0;
		DustVelocityCPU[i].z = 0.0;
		DustVelocityCPU[i].w = 0.0; // The w component of velocity holds the diameter of the dust particle.
		
		DustForceCPU[i].x = 0.0;
		DustForceCPU[i].y = 0.0;
		DustForceCPU[i].z = 0.0;
		DustForceCPU[i].w = 0.0; // The w component of the force holds the mass on the dust particle.
		
		IonWakeCPU[i].x = 0.0;
		IonWakeCPU[i].y = 0.0;
		IonWakeCPU[i].z = 0.0;
		IonWakeCPU[i].w = 0.0;  // The w component of the ionWake holds the charge.
		
		DustParentChildCPU[i].numberOfParents = 0;
		for(int j = 0; j < MAX_NUMBER_CHILDREN; j++)
		{
			DustParentChildCPU[i].childId[j] = -1;
		}
		
		DustColor[i].x = 0.0;
		DustColor[i].y = 0.0;
		DustColor[i].z = 0.0;
		DustColor[i].w = 0.0;
	}
	
	// Setting the dust color
	for(int i = 0; i < NumberOfDustParticles; i++)
	{
		// A nice brown is 0.707 0.395 0.113
		DustColor[i].x = 0.707;
		DustColor[i].y = 0.395;
		DustColor[i].z = 0.113;
		DustColor[i].w = 0.0;
	}
	
	// Setting the dust diameter, which then sets the dust charge and mass
	for(int i = 0; i < NumberOfDustParticles; i++)
	{
		test = 0;
		while(test ==0)
		{	
			// Getting two uniform random numbers in [0,1]
			temp1 = ((double) rand() / (RAND_MAX));
			temp2 = ((double) rand() / (RAND_MAX));
			
			// Getting ride of the end points so now all random numbers are in (0,1)
			if(temp1 == 0 || temp1 == 1 || temp2 == 0 || temp2 == 1) 
			{
				test = 0;
			}
			else
			{
				// Using Box-Muller to get a standard normally distributed random numbers 
				// from two uniformlly distributed random numbers.
				randomNumber = cos(2.0*PI*temp2)*sqrt(-2 * log(temp1));
				
				// Log normal
				if (DustDistributionType == 0)
				{
					randomNumber = exp(randomNumber);
					diameter = BaseDustDiameter + DustDiameterStandardDeviation*randomNumber;
					test = 1;
				}
				// Truncated normal distribution
				else if (DustDistributionType == 1)
				{
					diameter = BaseDustDiameter + DustDiameterStandardDeviation*randomNumber;
					if(((BaseDustDiameter - 2.0*DustDiameterStandardDeviation) <= diameter) 
							&& (diameter <= (BaseDustDiameter + 2.0*DustDiameterStandardDeviation))
							&& 0.0 < BaseDustDiameter)
					{
						test = 1;
					}
					else
					{
						test = 0;
					}
				}
				else
				{
				 	printf("\n Bad DustDistributionType: %d.  Should be 0 or 1. \n", DustDistributionType);
    					exit(0);
				}
			}
		}
		
		radius = diameter/2.0;
		
		// Now the mass of this dust particle will be determined off of its diameter and its density (at present all the dust particles have the same density).
		mass = DustDensity*(4.0/3.0)*PI*radius*radius*radius;
		
		// Setting the charge on the dust grain.
		numberOfElectrons = round(BaseElectronsPerUnitDiameter*diameter);
		charge = numberOfElectrons*ElectronCharge;
		
		DustPositionCPU[i].w = charge;
		DustVelocityCPU[i].w = diameter;
		DustForceCPU[i].w = mass;
	}
	
	// Just printing out the dust mean and standard deviation for a sanity check.
	double mean, std;
	mean = 0.0;
	for(int i = 0; i < NumberOfDustParticles; i++)
	{
		mean += DustVelocityCPU[i].w*LengthUnit*1000000.0;
	}
	mean /= NumberOfDustParticles;
	
	std = 0.0;
	for(int i = 0; i < NumberOfDustParticles; i++)
	{
		std += (DustVelocityCPU[i].w*LengthUnit*1000000.0 - mean)*(DustVelocityCPU[i].w*LengthUnit*1000000.0 - mean);
	}
	std /= NumberOfDustParticles;
	std = sqrt(std);
	printf("\n Dust Mean = %f in microns, Standard Deviation = %f", mean , std);
	
	// Setting the initial dust positions.
	for(int i = 0; i < NumberOfDustParticles; i++)
	{
		test = 0;
		while(test == 0)
		{
			// Get random number between -1 at 1.
			DustPositionCPU[i].x = ((float)rand()/(float)RAND_MAX)*2.0 - 1.0;
			DustPositionCPU[i].z = ((float)rand()/(float)RAND_MAX)*2.0 - 1.0;
			// Setting it to have random radius from 0 to radius of the radius of RadialConfinementScale.
			mag = sqrt(DustPositionCPU[i].x*DustPositionCPU[i].x + DustPositionCPU[i].z*DustPositionCPU[i].z);
			radius = ((float)rand()/(float)RAND_MAX)*RadialConfinementScale;
			if(0.0 < mag)
			{
				DustPositionCPU[i].x *= radius/mag;
				DustPositionCPU[i].z *= radius/mag;
			}
			else
			{
				DustPositionCPU[i].x = 0.0;
				DustPositionCPU[i].z = 0.0;
			}
			DustPositionCPU[i].y = ((float)rand()/(float)RAND_MAX)*RadialConfinementHeight/10.0 + RadialConfinementHeight - RadialConfinementHeight/10.0;
			test = 1;
			
			for(int j = 0; j < i; j++)
			{
				seperation = sqrt((DustPositionCPU[i].x-DustPositionCPU[j].x)*(DustPositionCPU[i].x-DustPositionCPU[j].x) + (DustPositionCPU[i].y-DustPositionCPU[j].y)*(DustPositionCPU[i].y-DustPositionCPU[j].y) + (DustPositionCPU[i].z-DustPositionCPU[j].z)*(DustPositionCPU[i].z-DustPositionCPU[j].z));
				if(seperation < 1.5*IonDebyeLength)
				{
					test = 0;
					break;
				}
			}
		}
	}

	// Setting the initial ionwake point change location and fraction of its parent's charge. 
	// We will use the .w to hold the fractional charge.
	for(int i = 0; i < NumberOfDustParticles; i++)
	{
		IonWakeCPU[i].x = 0.0;
		IonWakeCPU[i].y = -BaseIonWakeLength;
		IonWakeCPU[i].z = 0.0;
		// IonsWakes are positive and dust grains are negatively charged so you need the negative.
		IonWakeCPU[i].w = BaseIonWakeFractionalCharge*(-DustPositionCPU[i].w); 
	}
	
	printf("\n\n ********************************************************************************");
	printf("\n Initial conditions have been set");
	printf("\n ********************************************************************************\n");
}

void drawPicture()
{
	if(Trace == 0)
	{
		glClear(GL_COLOR_BUFFER_BIT);
		glClear(GL_DEPTH_BUFFER_BIT);
	}
	
	float dustSize = 0.1;
	
	for(int i = 0; i < NumberOfDustParticles; i++)
	{
		// Dust
		glColor3d(DustColor[i].x,DustColor[i].y,DustColor[i].z);
		glPushMatrix();
			glTranslatef(DustPositionCPU[i].x, DustPositionCPU[i].y, DustPositionCPU[i].z);
			glutSolidSphere(dustSize,5,5);
		glPopMatrix();	
		
		// IonWake
		glColor3d(1.0,0.0,0.0);
		glPushMatrix();
			glTranslatef(DustPositionCPU[i].x + IonWakeCPU[i].x, DustPositionCPU[i].y + IonWakeCPU[i].y, DustPositionCPU[i].z + IonWakeCPU[i].z);
			// Setting the size of the ionWake by the ratio of its chage to that of the dust's
			glutSolidSphere(dustSize*IonWakeCPU[i].w/(-DustPositionCPU[i].w), 10, 10);
		glPopMatrix();	
		
		// I added this in, not sure if it's better or not.
		/*glHint(GL_LINE_SMOOTH_HINT,GL_NICEST);
		glEnable(GL_LINE_SMOOTH);*/
		
		// This handles whether you want to draw lines between dust and companions.
		if(DustViewingAids)
		{
			// Line between Dust and its IonWake
			glLineWidth(1.0);
			glColor3d(0.0,0.0,1.0);
			glBegin(GL_LINES);
				glVertex3f(DustPositionCPU[i].x, DustPositionCPU[i].y, DustPositionCPU[i].z);
				glVertex3f(DustPositionCPU[i].x + IonWakeCPU[i].x, DustPositionCPU[i].y + IonWakeCPU[i].y, DustPositionCPU[i].z + IonWakeCPU[i].z);
			glEnd();
		
			// This handles drawing lines between dust and children.
			glLineWidth(0.75);
			for (int j = 0; j < MAX_NUMBER_CHILDREN; j++)
			{
				if(DustParentChildCPU[i].childId[j] != -1) 
				{
					glEnable(GL_LINE_SMOOTH);
					glEnable(GL_BLEND);
					glBegin(GL_LINES);				
					int compId = DustParentChildCPU[i].childId[j];
					// Red from parent to child.
					glColor3f(1.0f, 0.0f, 0.0f);
					glVertex3f(DustPositionCPU[i].x, DustPositionCPU[i].y, DustPositionCPU[i].z);
					glColor3f(0.0f, 1.0f, 0.0f);
					glVertex3f(DustPositionCPU[compId].x, DustPositionCPU[compId].y, DustPositionCPU[compId].z);
					glEnd();
				}
			}
			
		}
	}
	
	if(RadialConfinementViewingAids == 1)
	{
		glLineWidth(1.0);
		float divitions = 60.0;
		float angle = 2.0*PI/divitions;
		
		// Drawing the top of RadialConfinement ring.
		for(int i = 0; i < divitions; i++)
		{
			if(i < divitions/2) glColor3d(1.0,0.0,0.0);
			else glColor3d(0.0,0.0,1.0);
			glBegin(GL_LINES);
				glVertex3f(sin(angle*i)*RadialConfinementScale, RadialConfinementHeight, cos(angle*i)*RadialConfinementScale);
				glVertex3f(sin(angle*(i+1))*RadialConfinementScale, RadialConfinementHeight, cos(angle*(i+1))*RadialConfinementScale);
			glEnd();
		}
		
		// Drawing top of sheath ring.
		glColor3d(0.0,1.0,0.0);
		for(int i = 0; i < divitions; i++)
		{
			glBegin(GL_LINES);
				glVertex3f(sin(angle*i)*RadialConfinementScale, SheathHeight, cos(angle*i)*RadialConfinementScale);
				glVertex3f(sin(angle*(i+1))*RadialConfinementScale, SheathHeight, cos(angle*(i+1))*RadialConfinementScale);
			glEnd();
		}
		
		// Drawing the bottom plate.
		glColor3d(1.0,1.0,1.0);
		for(int i = 0; i < divitions; i++)
		{
			glBegin(GL_LINES);
				glVertex3f(sin(angle*i)*RadialConfinementScale, 0.0, cos(angle*i)*RadialConfinementScale);
				glVertex3f(sin(angle*(i+1))*RadialConfinementScale, 0.0, cos(angle*(i+1))*RadialConfinementScale);
			glEnd();
		}
			
		// Drawing a CutOffMultiplier*IonDebyeLength so we can get a prospective.
		glColor3d(0.0,1.0,0.0);
		glBegin(GL_LINES);
				glVertex3f(DustPositionCPU[0].x, DustPositionCPU[0].y, DustPositionCPU[0].z);
				glVertex3f(DustPositionCPU[0].x + CutOffMultiplier*IonDebyeLength, DustPositionCPU[0].y, DustPositionCPU[0].z);
		glEnd();
	}
	glutSwapBuffers();
	
	// Making a video of the run.
	if(MovieOn == 1)
	{
		glReadPixels(5, 5, XWindowSize, YWindowSize, GL_RGBA, GL_UNSIGNED_BYTE, Buffer);
		fwrite(Buffer, sizeof(int)*XWindowSize*YWindowSize, 1, MovieFile);
	}
}

// This kernel finds the total number of dust below each dust within a given cutoff range.
// The children will be ordered from nearest to farthest. To not have to dynamically allocate memory,
// we set a maximum number of children (MAX_NUMBER_CHILDREN) for each dust. 
// We use the dustParentChild structure to store parent-child information. In the event that two dust
// grains are within the cutoff range but at equal heights we choose the dust with a smaller index to 
// be the parent (see more details below).
// Also, before grouping dust, we reset the dustParentChild structure information to default values.
__global__ void groupDust(float4 *dustPos, float cutOff, float4 *ionWake, dustParentChildStructure *dustParentChildGPU, int numberOfParticles, float baseIonWakeLength, float baseIonWakeFractionalCharge)
{
	int myId = threadIdx.x + blockDim.x*blockIdx.x;
	
	int yourId, minId;
	float dx, dy, dz, d2, d, minDustDis;
	
	// Make sure we don't work past number of particles. May have more workers than dust grains.
	if (!(myId < numberOfParticles)) return;
	
	// Resetting dustParentChild structure information.
	// Set the number of parents to 0.
	dustParentChildGPU[myId].numberOfParents = 0;
	// Set every childId back to -1.
	for(int j = 0; j < MAX_NUMBER_CHILDREN; j++)
	{
		dustParentChildGPU[myId].childId[j] = -1;
	}
	// Reset the ionWake info back to 0.
	ionWake[myId].x = 0.0;
	ionWake[myId].y = -baseIonWakeLength;
	ionWake[myId].z = 0.0;
	// IonsWakes are positive and dust grains are negatively charged so you need the negative.
	// Set the ionWake charge back to a fraction of its parent dust's charge.
	ionWake[myId].w = baseIonWakeFractionalCharge*(-dustPos[myId].w);
	
	// Reading in global variables into local variables. This is because these variables are used
	// several times and reading from global memory is slow.
	float posMeX = dustPos[myId].x;
	float posMeY = dustPos[myId].y;
	float posMeZ = dustPos[myId].z;
	
	// Looping through to find all possible children.
	for (int k = 0; k < MAX_NUMBER_CHILDREN; k++)
	{
		minDustDis = cutOff;
		minId = -1;

		// Looping through all the dust grains.

		// Looping through each block on the grid.
		for(int j = 0; j < gridDim.x; j++)
		{
			// Looping through each thread i on the block j.
			for(int i = 0; i < blockDim.x; i++)	
			{
				yourId = i + blockDim.x*j;

				// Making sure we are not working on ourself and not past the number of particles.
				if((myId != yourId) && (yourId < numberOfParticles))
				{						
					dy = dustPos[yourId].y - posMeY;
					
					if (dy > 0.0f) continue; // must be below us to be a child.

					dx = dustPos[yourId].x - posMeX;
					dz = dustPos[yourId].z - posMeZ;

					// We can speed this up by not doing square root here.   ????????????????
					// Use d2 and cutoff squared, and calculate minDustDis2. ????????????????
					d2  = dx*dx + dy*dy + dz*dz; 
					d  = sqrt(d2);

					// If within the cutoff and below the me dust
					if (d <= cutOff)
					{
						// If you dust and me dust are at equal heights and you dust is smaller id, do nothing.
						// Let bigger id dusts be children if they are at equal heights.
						if(dy == 0.0f && yourId < myId)
						{ 
							continue;
						}
						if (d < minDustDis)
						{
							// Now we will make sure yourId is not already a child.
							int test = 1;
							for (int m = 0; m < k; m++)
							{
								if (dustParentChildGPU[myId].childId[m] == yourId) 
								{
									test = 0; // Set to 0 if we find a child, and break
									break;
								}
							}
							if (test == 1)
							{
								minId = yourId;
								minDustDis = d;
							}
						}							
					}	
					
				}
			}
		}
		
		if(minId != -1) // If something is found adjust the parent child structure.
		{
			dustParentChildGPU[myId].childId[k] = minId;
			atomicAdd(&(dustParentChildGPU[minId].numberOfParents), 1);
		}
		else // Potentially save a lot of iterations.
		{
			break; // if there are no children w/in cutoff here, there won't be in next.
		}
	}
	
}

/*
	This kernel adjusts the ion wake charge and length for the parent dust
	and the charge for all ion wakes of the children.
	If this function were disabled, you essentially would be incorporating the
	standard point charge model.
*/
__global__ void adjustPointCharge(float4 *dustPos, float cutOff, float4 *ionWake, 
								dustParentChildStructure *dustParentChildGPU, int *DustUsedCountGPU, 
								float chargeExchangeFraction, int numberOfParticles)
{
	float linearReduction, quadraticReduction, angularAdjustment, temp;
	int parentId, childId;
	float dx, dy, dz, d2, d;
	float dh, lx, lz;
	float addAmount;
	float inheritanceProportions[MAX_NUMBER_CHILDREN];
	
	parentId = threadIdx.x + blockDim.x*blockIdx.x;
	
	// Length (positive value) ion wake is below parent dust.
	float ionWakeLength = -ionWake[parentId].y;
	// Charge of parent dust's ion wake.
	float initialCharge = ionWake[parentId].w;
	float ionWakeCharge = ionWake[parentId].w;

	// If we are past the number of particles, this worker thread should do nothing.
	if(numberOfParticles <= parentId) return;
	// You can only give away inheritance if you have recieved all your inheritance 
	// from your parents. This is true if our number of parents is 0.
	if(dustParentChildGPU[parentId].numberOfParents != 0) return;
	
	// Making sure the parent dust doesn't give inheritance more than once.
	// Hense we set it's number of parents to -1 indicate that it has already been used.
	dustParentChildGPU[parentId].numberOfParents = -1;
	
	//totalInheritance = 0.0f; 
	
	float total = 0.0f;
	
	for (int k = 0; k < MAX_NUMBER_CHILDREN; k++)
	{
		childId = dustParentChildGPU[parentId].childId[k];

		// If the childId is -1, we can safely break because every id after that will equal -1.
		if (childId == -1) break;

		// Find the distance between this child dust and the parent dust.
		dx = dustPos[childId].x - dustPos[parentId].x;
		dy = dustPos[childId].y - dustPos[parentId].y;
		dz = dustPos[childId].z - dustPos[parentId].z;
		d2  = dx*dx + dy*dy + dz*dz;
		d  = sqrt(d2);

		// Producing a linear function that goes from 1 to 0 as the dust-dust distance, d, goes from cutOff to zero.
		// The linear will be used to reduce the distance the top point charge is below its dust.
		linearReduction = d/cutOff;
		
		// We now create a function that goes from 1 to 0 as the two dust go from a vertical to horizontal alignment.
		// In other words, as the vertical distance, dy, decreases, the angularAdjustment decreases.
		// dy/d is the cosine of the angle off vertical. This would create a 50% reduction at 60 degrees.
		// But we would like to have a 50% deduction at 45 degrees which is cosine squared or (dy/d)^2.
		angularAdjustment = dy*dy/d2;

		// Producing a quadratic function that goes from 1 to 0 as the dust-dust distance, d, goes from cutOff to zero.
		// The quadratic will be used to reduce the top point charge and increase the bottom point charge.
		// We are using a second order for the charge because as the bottom dust gets close to the top 
		// dust it will be eating up a ring (second order) of ions that would have added to the ionwake.	
		quadraticReduction = linearReduction*linearReduction;
		
		// We take off of ionWakeLength what it currently is minus (1.0f - linearReduction) * angularAdjustment.
		// The (1.0f - linearReduction) goes from 0 to 1 as your dust-dust distance, d, goes from cutOff to zero.
		// So, the ionWakeLength will be decreased as the child dust gets closer to the parent dust.
		// angularAdjustment is unmodified, so it will go from 1 to 0 as you go from a vertical to horizontal alignment.
		// So the ionWakeLength will also be decreased as the child dust gets more vertically aligned with parent dust.
		ionWakeLength -= ionWakeLength * (1.0f - linearReduction) * angularAdjustment;
		
		// This calculates how much charge to give to the child.
		// The (1.0f - quadraticReduction) goes from 0 to 1 as the child dust gets farther away from parent dust.
		// So, that results in a reduction of what is subtracted from the initialCharge as the distance increases.
		// angularAdjustment will go from 1 to 0 as you go from a vertical to horizontal alignment.
		// The opposite, horizontal to vertical alignment, would be 0 to 1.
		// So the more vertical the child dust is, larger the reduction will be.
		// This makes sense because the parent dust's ion wake charge decreases in charge if a child dust gets more 
		// vertically below it and raises up.
		
		temp = ionWakeCharge * (1.0f - quadraticReduction) * angularAdjustment;
		ionWakeCharge -= temp;
		
		// Saving these values so we can assign the correct proportion of the inherited charge to each child. 
		inheritanceProportions[k] = temp;
		total += temp;

		// Decrease the child's number of parents, because it received its inheritance.
		// We will actually give it inheritance in the next for loop.
		atomicAdd(&(dustParentChildGPU[childId].numberOfParents), -1);

		// Now we handle a slight tilt-in towards the horizontal line betweent parent and child.
		dh = sqrt(dx*dx + dz*dz);
		
		// The 1/4 dh on the end can be changed to make the effect stronger or weaker.
		lx = (dx/dh) * (1.0f-quadraticReduction) * (dh/4.0f);
		lz = (dz/dh) * (1.0f-quadraticReduction) * (dh/4.0f);

		// These do not need an atomicAdd because the parent dust must have received all its
		// inheritance, meaning it was already adjusted by its parent(s). So, we should be able
		// to safely modify its ionwake's x and z without problems.
		ionWake[parentId].x += lx;
		ionWake[parentId].z += lz;

		// These are the children, which are currently receiving inheritance, so we need atomicAdd.
		atomicAdd(&(ionWake[childId].x), -lx);
		atomicAdd(&(ionWake[childId].z), -lz);
	} // end for, we looped through every child
	
	// Setting the parent dust's values.
	ionWake[parentId].y = -ionWakeLength; // Minus sign because it is below the dust.
	ionWake[parentId].w = ionWakeCharge;
	
	
	float finalCharge = ionWake[parentId].w;
	
	for (int k = 0; k < MAX_NUMBER_CHILDREN; k++)
	{
		childId = dustParentChildGPU[parentId].childId[k];

		// If the childId is -1, we can safely break because every id after that will equal -1.
		if (childId == -1 || total == 0.0f) break;
		
		addAmount = (initialCharge - finalCharge) * chargeExchangeFraction * (inheritanceProportions[k] / total);
		
		atomicAdd(&(ionWake[childId].w), addAmount);
	}
	
	// The current dust gave all its inheritance, add one to counter.
	atomicAdd(&(DustUsedCountGPU[0]), 1);
}

// This is the main n-body kernel. Its job is to calculate and sum all of the forces for each dust grain in each direction.
__global__ void getForces(float4 *dustPos, float4 *dustForce, float4 *ionWake, float baseDustDiameter, 
		          float coulombConstant, float ionDebyeLength, float radialConfinementScale, float radialConfinementStrength, 
			  float sheathHeight, float bottomPlateForcingParameter, float gravity, int numberOfParticles)
{
	float force; 
	float dx, dy, dz, d2, d;
	int yourId;

	// Description of coordinate system (when in side view):
	// Negative x: left
	// Positive x: right	
	// Negative y: down
	// Positive y: up	
	// Negative z: backwards
	// Positive z: forward

	// Positions are not changed here so it need not be carried forward but forces do need to be 
	// carried forward so it can be used in the move function.
	
	int myId = threadIdx.x + blockDim.x*blockIdx.x;
	
	if(myId < numberOfParticles) // Making sure we don't work past the number of particles.
	{
		// Setting up local variables we will need instead of constantly reading slow global variables.
		float posMeX = dustPos[myId].x;
		float posMeY = dustPos[myId].y;
		float posMeZ = dustPos[myId].z;
		float chargeMe = dustPos[myId].w;
		
		float forceMeX = 0.0f;
		float forceMeY = 0.0f;
		float forceMeZ = 0.0f;
		float massMe = dustForce[myId].w;
		
		// For the current dust, calculate forces from every other dust and their ion wakes.
		for(int j = 0; j < gridDim.x; j++)
		{
			for(int i = 0; i < blockDim.x; i++)	
		    {
		    	yourId = i + blockDim.x*j;
		    	// Making sure we are not working on ourself and not past the number of particles.
				if(myId != yourId && yourId < numberOfParticles) 
				{					
					// Force caused by other dust. 
					// For forces, we use a Yukawa force, which takes shielding by ions and electrons into account.
					dx = dustPos[yourId].x - posMeX;
					dy = dustPos[yourId].y - posMeY;
					dz = dustPos[yourId].z - posMeZ;
					d2  = dx*dx + dy*dy + dz*dz;// + EPSILON2; // Added the 0.000001 to prevent singularities.
					d  = sqrt(d2);
					
					// In our units, Coulomb's constant is 1--therefore, it could be removed, however we leave it in for clarity.
					// dustPos.w holds the charge.
					force = (-coulombConstant*dustPos[yourId].w*chargeMe/d2)*(1.0f + d/(ionDebyeLength))*exp(-d/(ionDebyeLength)); 
					forceMeX += force*dx/d;
					forceMeY += force*dy/d;
					forceMeZ += force*dz/d;

					// Force caused by every other ionWake (point charge) to the me dust.
					dx = (dustPos[yourId].x + ionWake[yourId].x) - posMeX;
					dy = (dustPos[yourId].y + ionWake[yourId].y) - posMeY;
					dz = (dustPos[yourId].z + ionWake[yourId].z) - posMeZ;
					d2  = dx*dx + dy*dy + dz*dz;// + EPSILON2;  // Added the 0.000001 to prevent singularities.
					d  = sqrt(d2);
					
					// In our units, Coulomb's constant is 1--therefore, it could be removed, however we leave it in for clarity. 
					force = (-coulombConstant*ionWake[yourId].w*chargeMe/d2)*(1.0f + d/(ionDebyeLength))*exp(-d/(ionDebyeLength));
					forceMeX += force*dx/d;
					forceMeY += force*dy/d;	
					forceMeZ += force*dz/d;
				}
			}
		}
		// This flag indicates whether to calculate a force caused by your own ion wake.
		// This causes a lot of problems when turned on---turn on at your own risk.
		int useSelfPointCharge = 0;
		if(useSelfPointCharge == 1)
		{
			float ionWakeLengthMeX = ionWake[myId].x;
			float ionWakeLengthMeY = ionWake[myId].y;
			float ionWakeLengthMeZ = ionWake[myId].z;
			float ionWakeChargeMe = ionWake[myId].w;
			d2 = ionWakeLengthMeX*ionWakeLengthMeX + ionWakeLengthMeY*ionWakeLengthMeY + ionWakeLengthMeZ*ionWakeLengthMeZ;// + EPSILON2;
			d  = sqrt(d2);

			// In our units, Coulomb's constant is 1--therefore, it could be removed, however we leave it in for clarity. 
			force = (-coulombConstant*chargeMe*ionWakeChargeMe/d2)*(1.0f + d/(ionDebyeLength*10.0f))*exp(-d/(ionDebyeLength*10.0f));
			forceMeX += force*ionWakeLengthMeX/d;
			forceMeY += force*ionWakeLengthMeY/d;
			forceMeZ += force*ionWakeLengthMeZ/d;
		}
		
		// Getting dust to bottom plate force.
		// e field is bottomPlateForcingParameter*(posMeY - sheathHeight). 
		// This is the linear force that starts at the sheath. We got this from Dr. Mathews.
		if(posMeY < sheathHeight)
		{
			forceMeY += chargeMe*bottomPlateForcingParameter*(posMeY - sheathHeight);
		}
		
		// Getting push back from the radial Confinement region. Height is irrelevant.
		d  = sqrt(posMeX*posMeX + posMeZ*posMeZ);// + EPSILON;
		// If it is zero nothing needs to be done.
		if(d != 0.0) 
		{
			force = chargeMe*radialConfinementStrength*pow(d/radialConfinementScale,12.0);
			forceMeX += force*posMeX/d;
			forceMeZ += force*posMeZ/d;
		}
		
		// Getting force of gravity
		forceMeY += -gravity*massMe;
		
		// All the forces have been summed up for the dust grain so load them up to carry forward to the move function.
		// The mass was not changed so it doesn't need to be updated.
		dustForce[myId].x = forceMeX;
		dustForce[myId].y = forceMeY;
		dustForce[myId].z = forceMeZ;
	}
}
		
// This kernel has two main tasks: 
// The first task is to apply the leapfrog formulas to move the dust forward in time.
// The second task is to stochastically adjust the number of electrons attached to each dust grain.
__global__ void moveDust(float4 *dustPos, float4 *dustVel, float4 *dustForce, float drag, float electronCharge, 
						float baseDustDiameter, float baseElectronsPerUnitDiameter, float electronStdPerUnitDiameter, float sheathHeight, float dt, 
						float time, int numberOfDustParticles)
{
	curandState state;
	float randomNumber;
	int id = threadIdx.x + blockDim.x*blockIdx.x;
	
	if (id < numberOfDustParticles)
	{
		// Using the leapfrog formulas to move the dust forward in time.
		float diameter = dustVel[id].w; // dustVel.w holds the diameter of the dust.
		float invMass = 1.0f / dustForce[id].w; // dustForce.w is mass of the dust.
		float charge = dustPos[id].w; // dustPos.w holds the charge of the dust.

		// diameter^2 = (2*radius)^2 = 4*radius^2
		// (1/4)*drag*diameter^2 = drag*radius^2
		float dragCoefficient = drag * diameter * diameter * 0.25f;

		// When time is 0 in leapfrog, move the velocities forward a half time step.
		if(time == 0.0)
		{
			dustVel[id].x += 0.5f*dt*(dustForce[id].x - dragCoefficient*dustVel[id].x) * invMass;
			dustVel[id].y += 0.5f*dt*(dustForce[id].y - dragCoefficient*dustVel[id].y) * invMass;
			dustVel[id].z += 0.5f*dt*(dustForce[id].z - dragCoefficient*dustVel[id].z) * invMass;
		}
		else
		{
			dustVel[id].x += dt*(dustForce[id].x - dragCoefficient*dustVel[id].x) * invMass;
			dustVel[id].y += dt*(dustForce[id].y - dragCoefficient*dustVel[id].y) * invMass;
			dustVel[id].z += dt*(dustForce[id].z - dragCoefficient*dustVel[id].z) * invMass;
		}

		dustPos[id].x += dustVel[id].x*dt;
		dustPos[id].y += dustVel[id].y*dt;
		dustPos[id].z += dustVel[id].z*dt;
		
		// Randomly perturbating the dust electron count. 
		// This gets a little involved. I first get a standard normal distributed number (Mean 0 StDev 1).
		// Then I set its StDev to the number of electrons that fluctuate per unit dust diameter for this dust grain size.
		// Then I set the mean to how much above or below the base electron per unit dust size.
		// ie. if it has more than it should it has a higher prob of losing and vice versa if it has less than it should.
		// This is just what I came up with and it could be wrong but below is how I did this.
		// dustPos.w carries the charge and dustVel.w carries the diameter.

		// Initailizing the cudarand function.
		curand_init(clock64(), id, 0, &state); // TODO: Read up on curand() function
		// This gets a random number with mean 0.0 and stDev 1.0;.
		randomNumber = curand_normal(&state);
		// This sets the electron fluctuation for this sized dust grain and makes it the stDev.
		randomNumber *= electronStdPerUnitDiameter*diameter;
		
		// This has a mean of zero which would just create a random walk. I don't think this is what you want.
		// Dust grains with more electrons than they should have should in general loose electrons 
		// and those with less than they should should in general gain more electrons.
		// We will accomplish this by setting the mean to be the oposite of how much above or below 
		// the base amount you are at this time.
		// This works out to be base number - present number
		// baseElectronsPerUnitDiameter*diameter = base # of electrons
		// charge/electronCharge = # of electrons
		randomNumber += baseElectronsPerUnitDiameter*diameter - charge/electronCharge;
		
		// Now add/subtract this number of electron to the existing charge.
		dustPos[id].w += randomNumber*electronCharge;
	
		// If the amount of charge ends up being negative which probablistically it could, set it to zero
		if(dustPos[id].w > 0.0) dustPos[id].w = 0.0;

		// If the dust grain gets too close or passes through the floor. 
		// I put it at the top of the sheath, set its force to zero and set its mass, 
		// charge and diameter to the base (maybe it was too heavy).
		if(dustPos[id].y < 0.001f)
		{
			dustPos[id].y = sheathHeight;
			dustPos[id].w = 1.0; // The base charge should be 1

			dustVel[id].x = 0.0;
			dustVel[id].y = 0.0;

			dustVel[id].z = 0.0;
			dustVel[id].w = baseDustDiameter;
			
			dustForce[id].x = 0.0;
			dustForce[id].y = 0.0;
			dustForce[id].z = 0.0;
			dustForce[id].w = 1.0; // The base mass should be 1
			
			printf("\n myId = %d posMeX = %f posMeY = %f posMeZ = %f MassMe = %f, chargeMe = %f\n", id, dustPos[id].x, dustPos[id].y, dustPos[id].z, dustForce[id].w, dustPos[id].w);
		}
	}				
}

// This is basically our render loop (or a wrapper of sorts).
void n_body()
{	
	float cutOff = IonDebyeLength*CutOffMultiplier;
	if(Pause != 1)
	{	
		groupDust<<<Grid, Block>>>(DustPositionGPU, cutOff, IonWakeGPU, DustParentChildGPU, NumberOfDustParticles, BaseIonWakeLength, BaseIonWakeFractionalCharge);
		cudaDeviceSynchronize();
		
		// TODO: Do this in a smarter way.
		DustUsedCountCPU[0] = 0;
		cudaMemcpy( DustUsedCountGPU, DustUsedCountCPU, 1*sizeof(int), cudaMemcpyHostToDevice);
		errorCheck("cudaMemcpy DustUsedCountGPU up");
		
		while(DustUsedCountCPU[0] < NumberOfDustParticles)
		{
			adjustPointCharge<<<Grid, Block>>>(DustPositionGPU, cutOff, IonWakeGPU, DustParentChildGPU, DustUsedCountGPU, ChargeExchangeFraction, NumberOfDustParticles);
			cudaDeviceSynchronize();
			cudaMemcpy( DustUsedCountCPU, DustUsedCountGPU, 1*sizeof(int), cudaMemcpyDeviceToHost);
			errorCheck("cudaMemcpy DustUsedCountGPU up");
		}

		getForces<<<Grid, Block>>>(DustPositionGPU, DustForceGPU, IonWakeGPU, BaseDustDiameter, CoulombConstant, IonDebyeLength, RadialConfinementScale, RadialConfinementStrength, SheathHeight, BottomPlateForcingParameter, Gravity, NumberOfDustParticles);
		cudaDeviceSynchronize();

		if(Freeze == 0)
		{
			moveDust<<<Grid, Block>>>(DustPositionGPU, DustVelocityGPU, DustForceGPU, Drag, 
					ElectronCharge, BaseDustDiameter, BaseElectronsPerUnitDiameter, electronStdPerUnitDiameter, 
					SheathHeight, Dt, RunTime, NumberOfDustParticles);
			cudaDeviceSynchronize();
		}
		
				
		cudaMemcpy( DustPositionCPU, DustPositionGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
		errorCheck("cudaMemcpy DustPositionCPU down");
		cudaMemcpy( IonWakeCPU, IonWakeGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
		errorCheck("cudaMemcpy IonWakeCPU down");
		cudaMemcpy( DustParentChildCPU, DustParentChildGPU, NumberOfDustParticles*sizeof(dustParentChildStructure), cudaMemcpyDeviceToHost);
		errorCheck("cudaMemcpy DustParentChildCPU down");
		
		DrawTimer++;
		if(DrawTimer >= DrawRate) 
		{

			drawPicture();
			DrawTimer = 0;
		}

		PrintTimer++;
		if(PrintTimer >= PrintTimeRate) 
		{
			terminalPrint();
			PrintTimer = 0;
		}
		
		RunTime += Dt;
	}
	else
	{
		// This looks wierd but I had to do it so I could work on the view while it is paused. 
		// Consequence of the idle callback calling n_body().
		// drawPicture(); 
	}
}

/*
	This function is printed while the simulation runs to provide the user a list of the controls.
	It prints out things such as how to move, change the plate power, pause the simulation, etc.
*/
void terminalPrint()
{
	system("clear");
	//printf("\033[0;34m"); // print in blue.
	//printf("\033[0;33m"); // yellow
	printf("\033[0m"); // back to white.
	printf("\n DrawRate = %d\n", DrawRate);
	printf(" (q)   Quits the simulation\n");
	printf(" (h)   Help\n");
	printf(" (m/M) MovieStart/MovieStop\n");
	printf(" (s)   Takes a screen shot of the simulation\n");
	printf("\n");
	printf(" (V/v) Bottom plate Forcing Parameter: %e\n", BottomPlateForcingParameter*(MassUnit)/(TimeUnit*TimeUnit*ChargeUnit));
	printf(" (C/c) Radial Confinement Strength: %e\n", RadialConfinementStrength*(MassUnit*LengthUnit)/(TimeUnit*TimeUnit*ChargeUnit));
	printf(" (P/p) Gas Pressure in millitorr: %e\n", GasPressure);
	printf(" (I/i) BaseIonWakeFractionalCharge: %f\n", BaseIonWakeFractionalCharge);
	printf("\n");
	printf(" (1/!) On/off x/y/z movement for selected dust grain.");
	if (SelectedDustMove) printf(" (XYZ movement is ON.)\n\n");
	else printf(" XYZ movement is OFF.\n\n");
	if (PreviouslySelectedGrain != -1)	
	{
		printf(" Previously selected dust grain (pink) Id = %d, Child[0] = %d\n", PreviouslySelectedGrain, DustParentChildCPU[PreviouslySelectedGrain].childId[0]);
		printf(" Dust position is: (%+2.10f, %+2.10f, %+2.10f), charge = %+2.10f\n", DustPositionCPU[PreviouslySelectedGrain].x, DustPositionCPU[PreviouslySelectedGrain].y, DustPositionCPU[PreviouslySelectedGrain].z, DustPositionCPU[PreviouslySelectedGrain].w);
		printf(" IonWake position: (%+2.10f, %+2.10f, %+2.10f), charge = %+2.10f\n", IonWakeCPU[PreviouslySelectedGrain].x, IonWakeCPU[PreviouslySelectedGrain].y, IonWakeCPU[PreviouslySelectedGrain].z, IonWakeCPU[PreviouslySelectedGrain].w);
		float ionWakeLengthMeX = IonWakeCPU[PreviouslySelectedGrain].x;
		float ionWakeLengthMeY = IonWakeCPU[PreviouslySelectedGrain].y;
		float ionWakeLengthMeZ = IonWakeCPU[PreviouslySelectedGrain].z;
		float d2 = ionWakeLengthMeX*ionWakeLengthMeX + ionWakeLengthMeY*ionWakeLengthMeY + ionWakeLengthMeZ*ionWakeLengthMeZ;// + EPSILON2;
		float d = sqrt(d2);
		printf(" Dust is %+2.10f away from its ionwake, that squared is %+2.10f\n", d, d2);
		float dh = sqrt(ionWakeLengthMeX*ionWakeLengthMeX + ionWakeLengthMeZ*ionWakeLengthMeZ);// + EPSILON;
		printf("dx = %f, dz = %f, dh = %f\n", ionWakeLengthMeX, ionWakeLengthMeZ, dh);
	}
	printf("\n");
	if (SelectedDustGrainId1 != -1)
	{
		printf(" Currently selected dust grain (blue) Id = %d, Child[0] = %d\n", SelectedDustGrainId1, DustParentChildCPU[SelectedDustGrainId1].childId[0]);
		printf(" Dust position is: (%+2.10f, %+2.10f, %+2.10f), charge = %+2.10f\n", DustPositionCPU[SelectedDustGrainId1].x, DustPositionCPU[SelectedDustGrainId1].y, DustPositionCPU[SelectedDustGrainId1].z, DustPositionCPU[SelectedDustGrainId1].w);
		printf(" IonWake position: (%+2.10f, %+2.10f, %+2.10f), charge = %+2.10f\n", IonWakeCPU[SelectedDustGrainId1].x, IonWakeCPU[SelectedDustGrainId1].y, IonWakeCPU[SelectedDustGrainId1].z, IonWakeCPU[SelectedDustGrainId1].w);
		float ionWakeLengthMeX = IonWakeCPU[SelectedDustGrainId1].x;
		float ionWakeLengthMeY = IonWakeCPU[SelectedDustGrainId1].y;
		float ionWakeLengthMeZ = IonWakeCPU[SelectedDustGrainId1].z;
		float ionWakeChargeMe = IonWakeCPU[SelectedDustGrainId1].w;
		float d2 = ionWakeLengthMeX*ionWakeLengthMeX + ionWakeLengthMeY*ionWakeLengthMeY + ionWakeLengthMeZ*ionWakeLengthMeZ;// + EPSILON2;
		float d = sqrt(d2);
		printf(" Dust is %+2.10f away from its ionwake, that squared is %+2.10f\n", d2, d);
		float dh = sqrt(ionWakeLengthMeX*ionWakeLengthMeX + ionWakeLengthMeZ*ionWakeLengthMeZ);// + EPSILON;
		printf("dx = %f, dz = %f, dh = %f\n", ionWakeLengthMeX, ionWakeLengthMeZ, dh);
			// In our units, Coulomb's constant is 1--therefore, it could be removed, however we leave it in for clarity. 
			float forceX = (-CoulombConstant*DustPositionCPU[SelectedDustGrainId1].w*ionWakeChargeMe/d2)*(1.0f + d/IonDebyeLength)*exp(-d/IonDebyeLength)*ionWakeLengthMeX/d;
			float forceY = (-CoulombConstant*DustPositionCPU[SelectedDustGrainId1].w*ionWakeChargeMe/d2)*(1.0f + d/IonDebyeLength)*exp(-d/IonDebyeLength)*ionWakeLengthMeY/d;
			float forceZ = (-CoulombConstant*DustPositionCPU[SelectedDustGrainId1].w*ionWakeChargeMe/d2)*(1.0f + d/IonDebyeLength)*exp(-d/IonDebyeLength)*ionWakeLengthMeZ/d;
			printf("Forces in x: %f, y:%f, z:%f", forceX, forceY, forceZ);
	}
	if (PreviouslySelectedGrain != -1 && SelectedDustGrainId1 != -1)
	{
		// Calculations are you - me, where you is previously selected dust and me is current selected dust.
		float dx = DustPositionCPU[PreviouslySelectedGrain].x - DustPositionCPU[SelectedDustGrainId1].x;
		float dy = DustPositionCPU[PreviouslySelectedGrain].y - DustPositionCPU[SelectedDustGrainId1].y;
		float dz = DustPositionCPU[PreviouslySelectedGrain].z - DustPositionCPU[SelectedDustGrainId1].z;
		float d2 = dx*dx + dy*dy + dz*dz;
		float d = sqrt(d2);
		printf("\nDistance between selected and previous dust = %f and cutoff = %f\n", d, CutOffMultiplier*IonDebyeLength);

	}
	
	printf(" (w) Top view/Side view toggle\n");
	printf(" (X/x) Move Right/Left\n");
	printf(" (Y/y) Move Up/Down\n");
	printf(" (Z/z) Move Out/In\n");
	printf("\n");
	
	printf(" (f) Freeze on/off toggle.             ");
	if (Freeze == 1) printf(" (Dust movement is OFF.)\n");
	else printf(" (Dust movement is ON.)\n");
	
	printf(" (r) Run/Pause toggle.                 ");
	if(Pause == 1) printf(" (The simulation is paused.)\n");
	else printf(" (The simulation is running.)\n");
	
	printf(" (t) Trace on/off toggle.              ");
	if(Trace == 1) printf(" (Tracing is ON.)\n");
	else printf(" (Tracing is OFF.)\n");
	
	printf(" (a) RC viewing aids on/off toggle.    ");
	if(RadialConfinementViewingAids == 1) printf(" (RC Viewing aids are ON.)\n");
	else printf(" (RC Viewing aids are OFF.)\n");
	
	printf(" (g) Dust Viewing aids on/off toggle.  ");
	if (DustViewingAids == 0) printf(" (Dust Viewing aids OFF)\n\n"); 
	else if (DustViewingAids == 1)	printf(" (Dust Viewing aids are ON)\n\n");
	else printf("DustViewingAids = %d\n", DustViewingAids);

	printf(" (9) Frustrum view\n");
	printf(" (0) Ortho View\n");
	
	printf("\n");
	printf(" Total run time = %f seconds\n", RunTime*TimeUnit);
	printf("\n");
	printf("\033[0m");
}

void setup()
{	
	readSimulationParameters();
	allocateMemory();
	setUnitConvertions();
	PutConstantsIntoOurUnits();
	setInitialConditions();
	
	// Coping up to GPU
	cudaMemcpy( DustPositionGPU, DustPositionCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
	errorCheck("cudaMemcpy DustPositionCPU up");
   	cudaMemcpy( DustVelocityGPU, DustVelocityCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
	errorCheck("cudaMemcpy DustVelocityGPU up");
	cudaMemcpy( DustForceGPU, DustForceCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
	errorCheck("cudaMemcpy DustForceGPU up");
	cudaMemcpy( IonWakeGPU, IonWakeCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
	errorCheck("cudaMemcpy IonWakeGPU up");
	cudaMemcpy( DustParentChildGPU, DustParentChildCPU, NumberOfDustParticles*sizeof(dustParentChildStructure), cudaMemcpyHostToDevice );
	errorCheck("cudaMemcpy DustParentChildGPU up");
    
	DrawTimer = 0;
	PrintTimer = 0;
	RunTime = 0.0;
	Pause = 1;
	MovieOn = 0;
	Trace = 0;
	View = 1;
	DrawBottomRing = 1;
	MouseOn = 0;
	RadialConfinementViewingAids = 1;
	SingleOrPairOfDust = 0;
	Freeze = 0; // start unfrozen.
	SelectedDustMove = 0;
	SelectedDustGrainId1 = -1;
	PreviouslySelectedGrain = -1;
	DustViewingAids = 1; // On by default.
	
	printf("\nThe simulation is paused. Press the r key to start.\n");
}

void errorCheck(const char *message)
{
  cudaError_t  error;
  error = cudaGetLastError();

  if(error != cudaSuccess)
  {
    printf("\n CUDA ERROR: %s = %s\n", message, cudaGetErrorString(error));
    exit(0);
  }
}

void processMainMenuEvents(int option);
void processSubMenuEvents(int option);
int mainMenu;
int unPauseMenu;
void createGLUTMenus() {
	mainMenu = glutCreateMenu(processMainMenuEvents);
	// attach the menu to the right button
	glutAttachMenu(GLUT_RIGHT_BUTTON);
	glutAddMenuEntry("Select One Dust Grain", ONE_GRAIN);
	glutAddMenuEntry("Select Two Dust Grains", TWO_GRAINS);
	glutAddMenuEntry("Start Movie", START_MOVIE);
	glutAddMenuEntry("Stop Movie", STOP_MOVIE);
	glutAddMenuEntry("Take Screenshot", TAKE_SCREENSHOT);
	glutAddMenuEntry("Help", HELP);
}

void processMainMenuEvents(int option) 
{
	switch (option) 
	{
		case ONE_GRAIN:
			KeyPressed('l', 0, 0);
			break;
		case TWO_GRAINS:
			KeyPressed('L', 0, 0);
			break;
		case START_MOVIE:
			KeyPressed('m', 0, 0);
			break;
		case STOP_MOVIE:
			KeyPressed('M', 0, 0);
			break;
		case TAKE_SCREENSHOT:
			KeyPressed('s', 0, 0);
			break;
		case HELP:
			KeyPressed('h', 0, 0);
			break;
	}
}

int main(int argc, char** argv)
{
	setup();
	
	XWindowSize = 1000;
	YWindowSize = 1000; 

	// Clip plains
	Near = 0.2;
	Far = 80.0*RadialConfinementScale;

	//Direction where your eye is located
	EyeX = 0.0*RadialConfinementScale;
	EyeY = 0.5*RadialConfinementHeight;
	EyeZ = 3.0*RadialConfinementScale;
	
	//EyeX = 0.01*RadialConfinementScale;
	//EyeY = RadialConfinementHeight;
	//EyeZ = 0.0*RadialConfinementScale;

	//Where you are looking
	CenterX = 0.0;
	CenterY = 0.5*RadialConfinementHeight;
	CenterZ = 0.0;
	
	//Keeoing track of where your center is as you move your view
	CenterOfView.x = CenterX;
	CenterOfView.y = CenterY;
	CenterOfView.z = CenterZ;

	//Up vector for viewing
	UpX = 0.0;
	UpY = 1.0;
	UpZ = 0.0;
	
	//Keeoing track of your angle of view as you move your view
	AngleOfView.x = 0.0;
	AngleOfView.y = 0.0;
	AngleOfView.z = 0.0;
	
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(5,5);
	Window = glutCreateWindow("Dust Crystal Viewer");
	
	gluLookAt(EyeX, EyeY, EyeZ, CenterX, CenterY, CenterZ, UpX, UpY, UpZ);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glFrustum(-0.2, 0.2, -0.2, 0.2, Near, Far);
	glMatrixMode(GL_MODELVIEW);
	glClearColor(0.0, 0.0, 0.0, 0.0);
	
	GLfloat light_position2[] = {1.0, 1.0, 0.0, 0.0};
	
	GLfloat light_position[] = {1.0, 1.0, 1.0, 1.0};
	GLfloat light_ambient[]  = {0.0, 0.0, 0.0, 1.0};
	GLfloat light_diffuse[]  = {1.0, 1.0, 1.0, 1.0};
	GLfloat light_specular[] = {1.0, 1.0, 1.0, 1.0};
	GLfloat lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0};
	GLfloat mat_specular[]   = {1.0, 1.0, 1.0, 1.0};
	GLfloat mat_shininess[]  = {10.0};
	glShadeModel(GL_SMOOTH);
	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);
	
	glLightfv(GL_LIGHT0, GL_POSITION, light_position2);
	glEnable(GL_LIGHT1);
	
	glutDisplayFunc(Display);
	glutReshapeFunc(reshape);
	glutMouseFunc(mymouse);
	glutKeyboardFunc(KeyPressed);
	glutIdleFunc(idle);
	createGLUTMenus();
	glutMainLoop();
	return 0;
}

