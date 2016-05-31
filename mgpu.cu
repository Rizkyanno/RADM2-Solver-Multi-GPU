#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime_api.h>
#include <limits>

// Numerical constants
#define ZERO			(double)0.0
#define ONE			(double)1.0
#define TWO			(double)2.0
#define HALF			(double)0.5
#define THREE_HUNDRED		(double)300.0
#define DeltaMin		(double)1.0e-6

#define NSPEC                61          // Number of chemical species
#define NVAR                 59          // Number of Variable species
#define NVARACT              55          // Number of Active species
#define NFIX                 2           // Number of Fixed species
#define NREACT               156         // Number of reactions
#define NVARST               0           // Starting of variables in conc. vect.
#define NFIXST               59          // Starting of fixed in conc. vect.
#define NONZERO              572         // Number of nonzero entries in Jacobian
#define LU_NONZERO           659         // Number of nonzero entries in LU factoriz. of Jacobian
#define CNVAR                60          // (NVAR+1) Number of elements in compressed row format
#define NLOOKAT              0           // Number of species to look at
#define NMONITOR             0           // Number of species to monitor
#define NMASS                1           // Number of atoms to check mass balance

// molecules/mol
#define navgdro (6.022e23)

// molecular weights
#define mwh (1.0079)
#define mwo (15.9994)
#define mwair (28.97)
#define mwh2o (2*mwh+mwo)

// dens2con air
#define dens2con_a (1.0e-3*(1.0/mwair)*navgdro)

// dens2con water
#define dens2con_w (1.0e-3*(1.0/mwh2o)*navgdro)

#define NPHY			3

#define NRAD			29
#define NPH			28

// pointers to photolysis rates
#define Pj_o31d			0
#define Pj_o33p			1
#define Pj_no2			2
#define Pj_no3o2		3
#define Pj_no3o			4
#define Pj_hno2			5
#define Pj_hno3			6
#define Pj_hno4			7
#define Pj_h2o2			8
#define Pj_ch2or		9
#define Pj_ch2om		10
#define Pj_ch3cho		11
#define Pj_ch3coch3		12
#define Pj_ch3coc2h5		13
#define Pj_hcocho		14
#define Pj_ch3cocho		15
#define Pj_hcochest		16
#define Pj_ch3o2h		17
#define Pj_ch3coo2h		18
#define Pj_ch3ono2		19
#define Pj_hcochob		20
#define Pj_macr			21
#define Pj_n2o5			22
#define Pj_o2			23
#define Pj_pan			24
#define Pj_acet			25
#define Pj_mglo			26
#define Pj_hno4_2		27

#define P_o3 (5)
#define P_h2o2 (7)
#define P_no (4)
#define P_no2 (3)
#define P_no3 (17)
#define P_n2o5 (16)
#define P_hono (31)
#define P_hno3 (6)
#define P_hno4 (32)
#define P_so2 (1)
#define P_sulf (2)
#define P_co (23)
#define P_eth (22)
#define P_hc3 (19)
#define P_hc5 (20)
#define P_hc8 (21)
#define P_ol2 (24)
#define P_olt (25)
#define P_oli (26)
#define P_iso (39)
#define P_tol (27)
#define P_xyl (28)
#define P_csl (38)
#define P_hcho (9)
#define P_ald (8)
#define P_ket (33)
#define P_gly (34)
#define P_mgly (35)
#define P_dcb (36)
#define P_onit (37)
#define P_pan (18)
#define P_tpan (30)
#define P_op1 (10)
#define P_op2 (11)
#define P_paa (12)
#define P_ora1 (13)
#define P_ora2 (14)
#define P_HO (42)
#define P_ho2 (43)
#define P_aco3 (29)
#define P_ch4 (41)
#define P_co2 (40)

// Index declaration for variable species in VAR
#define ind_SULF             0
#define ind_ORA1             1
#define ind_ORA2             2
#define ind_CO2              3
#define ind_SO2              4
#define ind_ETH              5
#define ind_O1D              6
#define ind_HC5              7
#define ind_HC8              8
#define ind_TOL              9
#define ind_XYL              10
#define ind_TPAN             11
#define ind_HONO             12
#define ind_H2O2             13
#define ind_N2O5             14
#define ind_HC3              15
#define ind_CH4              16
#define ind_PAA              17
#define ind_O3P              18
#define ind_HNO4             19
#define ind_OP1              20
#define ind_CSL              21
#define ind_PAN              22
#define ind_OL2              23
#define ind_HNO3             24
#define ind_CO               25
#define ind_ISO              26
#define ind_OLT              27
#define ind_OLI              28
#define ind_DCB              29
#define ind_GLY              30
#define ind_XNO2             31
#define ind_KET              32
#define ind_MGLY             33
#define ind_TOLP             34
#define ind_XYLP             35
#define ind_OLTP             36
#define ind_OLN              37
#define ind_XO2              38
#define ind_OL2P             39
#define ind_HC5P             40
#define ind_OP2              41
#define ind_HCHO             42
#define ind_HC8P             43
#define ind_TCO3             44
#define ind_O3               45
#define ind_ONIT             46
#define ind_ALD              47
#define ind_OLIP             48
#define ind_KETP             49
#define ind_HO2              50
#define ind_MO2              51
#define ind_OH               52
#define ind_NO3              53
#define ind_ACO3             54
#define ind_HC3P             55
#define ind_ETHP             56
#define ind_NO               57
#define ind_NO2              58

#define ind_H2O              59
#define ind_M                60

#define indf_H2O             0
#define indf_M               1

#define MAX(a,b) ( ((a) >= (b)) ?(a):(b)  )
#define MIN(b,c) ( ((b) <  (c)) ?(b):(c)  )
#define ABS(x)   ( ((x) >=  0 ) ?(x):(-x) )
#define SQRT(d)  ( pow((d),0.5)  )
#define SIGN(x)  ( ((x) >=  0 ) ?[0]:(-1) )

int * dev_LU_IROW;                      // Row indexes of the LU Jacobian of variables
int * dev_LU_ICOL;                      // Column indexes of the LU Jacobian of variables
int * dev_LU_CROW;                      // Compressed row indexes of the LU Jacobian of variables

texture<int, 1, cudaReadModeElementType> tex_LU_DIAG;
texture<int, 1, cudaReadModeElementType> tex_LU_CROW;
texture<int, 1, cudaReadModeElementType> tex_LU_ICOL;
#define LU_DIAG(k) tex1Dfetch(tex_LU_DIAG, k);
#define LU_CROW(k) tex1Dfetch(tex_LU_CROW, k);
#define LU_ICOL(k) tex1Dfetch(tex_LU_ICOL, k);
int * dev_LU_DIAG; 

// Collect statistics: global variables
int Nfun,Njac,Nstp,Nacc,Nrej,Ndec,Nsol,Nsng;

static int * d_vecMask;
static int * d_vecMask2;
static int * d_RejectLastH;
static int * d_RejectMoreH;
static double * d_vecT;
static double * d_vecH;

double Tau;
double * Ynew;
double * Fcn;

__constant__ float const_ros_M[4];  // constant cache work with double?
__constant__ float const_ros_E[4];

static int first_accept_step = 1;

int i;

int device;
int startDevice = 0;
int endDevice = 2;

FILE *fptr1;
FILE *fptr2;
FILE *fptr3;
FILE *fptr4;
FILE *fptr5;
FILE *fptr6;

double value1;
double value2;
double value3;
double value4;
double value5;
double value6;

char line1[20];
char line2[20];
char line3[20];
char line4[20];
char line5[20];
char line6[20];

double wrfchemdata[3778560];
double wrfmoistdata[276480];
double wrfvdrog3data[391680];
double wrfphydata[138240];
double wrfraddata[391680];
double wrfphdata[1290240];

// Device data
typedef struct {
	double * d_VAR;
	double * d_FIX;
	double * d_RCONST;
	double * d_TEMP;
	double * d_CFACTOR;
	double * d_RC_N2O5;
	double * d_JV;

	double * d_wrf_chem;
	double * d_wrf_moist;
	double * d_wrf_vdrog3;
	double * d_wrf_phy;
	double * d_wrf_rad;
	double * d_wrf_ph;

	double * d_refwrfchemdata;
	double * d_refwrfmoistdata;
	double * d_refwrfvdrog3data;
	double * d_refwrfphydata;
	double * d_refwrfraddata;
	double * d_refwrfphdata;

	double * Fcn0;
	double * Jac0;

	cudaStream_t stream;
} WRFDataDevice;

WRFDataDevice DataDevice[2];

double tempFcn0[1705041];

double tempJac0[19044441];

// CUDA Timer
typedef struct {
	cudaEvent_t start;
	cudaEvent_t stop;
	float elapsed;
} cudaTimer;

void cudaStartTimer(cudaTimer *timer, int device, cudaStream_t stream) {
	cudaSetDevice(device);
	cudaEventCreate(&timer[0].start);
	cudaEventCreate(&timer[0].stop);
	cudaEventRecord(timer[0].start, stream);
}

void cudaStopTimer(cudaTimer *timer, int device, cudaStream_t stream) {
	cudaSetDevice(device);
	cudaEventRecord(timer[0].stop, stream);
	cudaEventSynchronize(timer[0].stop);
	cudaEventElapsedTime(&timer[0].elapsed, timer[0].start, timer[0].stop);
}

cudaTimer timingconvert;
cudaTimer timingupdaterconst;
cudaTimer timingfun;
cudaTimer timingjac;
cudaTimer timingludecomp;

void debug_dump_real_i(uint32_t len, double * dptr, int p, uint32_t ni, int stride ) {
        uint32_t i;

        uint32_t size = len*sizeof(double);
        double * tmp = (double*)malloc(size);

	cudaSetDevice(0);
        cudaMemcpy(tmp, dptr, size, cudaMemcpyDeviceToHost);

        fprintf(stderr,"\n");
        for(i = 0; i < ni; i++) {
                fprintf(stderr,"[%d]:\t%E\n", i, tmp[i*stride + p]);
        }
        fprintf(stderr,"\n");

        free(tmp);
}

void openingWRFChemData() {
	if ((fptr1 = fopen("/home/prafajar/mgpu/wrfchem.txt", "rt")) == NULL){
		printf("Error! opening file");
		exit(1);         // Program exits if file pointer returns NULL. 
	}
	while (fgets(line1, 3778560, fptr1) != NULL) {
		sscanf(line1, "%lf", &value1);
		wrfchemdata[i] = value1;
		i++;
	}
	i = 0;
	fclose(fptr1);

	if ((fptr2 = fopen("/home/prafajar/mgpu/wrfmoist.txt", "rt")) == NULL){
		printf("Error! opening file");
		exit(1);         // Program exits if file pointer returns NULL. 
	}
	while (fgets(line2, 276480, fptr2) != NULL) {
		sscanf(line2, "%lf", &value2);
		wrfmoistdata[i] = value2;
		i++;
	}
	i = 0;
	fclose(fptr2);

	if ((fptr3 = fopen("/home/prafajar/mgpu/wrfph.txt", "rt")) == NULL){
		printf("Error! opening file");
		exit(1);         // Program exits if file pointer returns NULL. 
	}
	while (fgets(line3, 1290240, fptr3) != NULL) {
		sscanf(line3, "%lf", &value3);
		wrfphdata[i] = value3;
		i++;
	}
	i = 0;
	fclose(fptr3);

	if ((fptr4 = fopen("/home/prafajar/mgpu/wrfphy.txt", "rt")) == NULL){
		printf("Error! opening file");
		exit(1);         // Program exits if file pointer returns NULL. 
	}
	while (fgets(line4, 138240, fptr4) != NULL) {
		sscanf(line4, "%lf", &value4);
		wrfphydata[i] = value4;
		i++;
	}
	i = 0;
	fclose(fptr4);

	if ((fptr5 = fopen("/home/prafajar/mgpu/wrfrad.txt", "rt")) == NULL){
		printf("Error! opening file");
		exit(1);         // Program exits if file pointer returns NULL. 
	}
	while (fgets(line5, 391680, fptr5) != NULL) {
		sscanf(line5, "%lf", &value5);
		wrfraddata[i] = value5;
		i++;
	}
	i = 0;
	fclose(fptr5);

	if ((fptr6 = fopen("/home/prafajar/mgpu/wrfvdrog3.txt", "rt")) == NULL){
		printf("Error! opening file");
		exit(1);         // Program exits if file pointer returns NULL. 
	}
	while (fgets(line6, 391680, fptr6) != NULL) {
		sscanf(line6, "%lf", &value6);
		wrfvdrog3data[i] = value6;
		i++;
	}
	i = 0;
	fclose(fptr6);

	printf("Opening WRF-Chem data\n\n");
}

void initializeWRFChemData() {
	int dom_dim = 28899;

	for(device = startDevice; device < endDevice; device++) {
		cudaSetDevice(device);
		cudaStreamCreate(&DataDevice[device].stream);
	}

	for(device = startDevice; device < endDevice; device++) {
		cudaSetDevice(device);
		cudaMallocHost((void**)&DataDevice[device].d_wrf_chem, 82 * 48 * 20 * 48 * sizeof(double));
		cudaMallocHost((void**)&DataDevice[device].d_wrf_moist, 6 * 48 * 20 * 48 * sizeof(double));
		cudaMallocHost((void**)&DataDevice[device].d_wrf_vdrog3, 17 * 48 * 20 * 48 * sizeof(double));
		cudaMallocHost((void**)&DataDevice[device].d_wrf_phy, 3 * 48 * 20 * 48 * sizeof(double));
		cudaMallocHost((void**)&DataDevice[device].d_wrf_rad, 29 * 48 * 20 * 48 * sizeof(double));
		cudaMallocHost((void**)&DataDevice[device].d_wrf_ph, 28 * 48 * 20 * 48 * sizeof(double));
	}

	for(device = startDevice; device < endDevice; device++) {
		cudaSetDevice(device);
	
		for(i = 0; i < 3778560; i++) {
			DataDevice[device].d_wrf_chem[i] = wrfchemdata[i];
		}

		for(i = 0; i < 276480; i++) {
			DataDevice[device].d_wrf_moist[i] = wrfmoistdata[i];
		}

		for(i = 0; i < 391680; i++) {
			DataDevice[device].d_wrf_vdrog3[i] = wrfvdrog3data[i];
		}

		for(i = 0; i < 138240; i++) {
			DataDevice[device].d_wrf_phy[i] = wrfphydata[i];
		}

		for(i = 0; i < 391680; i++) {
			DataDevice[device].d_wrf_rad[i] = wrfraddata[i];
		}

		for(i = 0; i < 1290240; i++) {
			DataDevice[device].d_wrf_ph[i] = wrfphdata[i];
		}
	}

	for(device = startDevice; device < endDevice; device++) {
		cudaSetDevice(device);
		
		cudaMalloc((void**)&DataDevice[device].d_VAR,     NVAR*dom_dim*sizeof(double));
		cudaMalloc((void**)&DataDevice[device].d_FIX,     NFIX*dom_dim*sizeof(double));
		cudaMalloc((void**)&DataDevice[device].d_RCONST,  NREACT*dom_dim*sizeof(double));
		cudaMalloc((void**)&DataDevice[device].d_TEMP,    dom_dim*sizeof(double));
		cudaMalloc((void**)&DataDevice[device].d_CFACTOR, dom_dim*sizeof(double));
		cudaMalloc((void**)&DataDevice[device].d_RC_N2O5, dom_dim*sizeof(double));
		cudaMalloc((void**)&DataDevice[device].d_JV,      NPH*dom_dim*sizeof(double));

		cudaMalloc((void**)&DataDevice[device].Fcn0,  852520 * sizeof(double));
		cudaMalloc((void**)&DataDevice[device].Jac0,  9522220 * sizeof(double));

		cudaMalloc((void**)&DataDevice[device].d_refwrfchemdata, (82 * 48 * 20 * 48) * sizeof(double));
		cudaMalloc((void**)&DataDevice[device].d_refwrfmoistdata, (6 * 48 * 20 * 48) * sizeof(double));
		cudaMalloc((void**)&DataDevice[device].d_refwrfvdrog3data, (17 * 48 * 20 * 48) * sizeof(double));
		cudaMalloc((void**)&DataDevice[device].d_refwrfphydata, (3 * 48 * 20 * 48) * sizeof(double));
		cudaMalloc((void**)&DataDevice[device].d_refwrfraddata, (29 * 48 * 20 * 48) * sizeof(double));
		cudaMalloc((void**)&DataDevice[device].d_refwrfphdata, (28 * 48 * 20 * 48) * sizeof(double));
	}

	for(device = startDevice; device < endDevice; device++) {
		cudaSetDevice(device);

		cudaMemcpyAsync(DataDevice[device].d_refwrfchemdata, DataDevice[device].d_wrf_chem, 
				82 * 48 * 20 * 48 * sizeof(double), cudaMemcpyHostToDevice, 
				DataDevice[device].stream);
		cudaMemcpyAsync(DataDevice[device].d_refwrfmoistdata,  DataDevice[device].d_wrf_moist,
				6 * 48 * 20 * 48 * sizeof(double), cudaMemcpyHostToDevice, 
				DataDevice[device].stream);
		cudaMemcpyAsync(DataDevice[device].d_refwrfvdrog3data, DataDevice[device].d_wrf_vdrog3, 
				17 * 48 * 20 * 48 * sizeof(double), cudaMemcpyHostToDevice, 
				DataDevice[device].stream);
		cudaMemcpyAsync(DataDevice[device].d_refwrfphydata, DataDevice[device].d_wrf_phy, 
				3 * 48 * 20 * 48 * sizeof(double), cudaMemcpyHostToDevice, 
				DataDevice[device].stream);
		cudaMemcpyAsync(DataDevice[device].d_refwrfraddata, DataDevice[device].d_wrf_rad, 
				29 * 48 * 20 * 48 * sizeof(double), cudaMemcpyHostToDevice, 
				DataDevice[device].stream);
		cudaMemcpyAsync(DataDevice[device].d_refwrfphdata, DataDevice[device].d_wrf_ph, 
				28 * 48 * 20 * 48 * sizeof(double), cudaMemcpyHostToDevice, 
				DataDevice[device].stream);
	}

	printf("Initializing WRF-Chem device data\n\n");
}

__global__ void dev_convert(
	double * VAR,      double * FIX,       double * TEMP,
	double * CFACTOR,  double * RC_N2O5,   double * jv,
	double * wrfphdata, double * wrfchemdata, double * wrfraddata, 
	double * wrfphydata, double * wrfmoistdata) {

	int i = threadIdx.x;
	int j = blockIdx.y;
	int k = blockIdx.x;
	int p = j*gridDim.x*blockDim.x + k*blockDim.x + i;

	double conv;
	double es, qvs, rh;

	#define MY_I(i) (i)
	// Adjust pointers to select desired cell
	VAR     = &VAR[p*NVAR];
	FIX     = &FIX[p*NFIX];
	jv      = &jv[p*NPH];

        // Adjust indexes for lower bound
        i += 4;
        j += 4;
        k += 0;

        // 3rd body concentration (molec/cm^3)
        FIX[MY_I(indf_M)] = dens2con_a * wrfphydata[((2)*48*20*48 + (j)*20*48 + (k)*48 + (i))];

        // water concentration (molec/cm^3)
        FIX[MY_I(indf_H2O)] = dens2con_w * (wrfmoistdata[((1)*48*20*48 + (j)*20*48 + (k)*48 + (i))]) * (wrfphydata[((2)*48*20*48 + (j)*20*48 + (k)*48 + (i))]);

        // temperature (K)
        *TEMP = wrfphydata[((1)*48*20*48 + (j)*20*48 + (k)*48 + (i))];

        // calculate rate constant for n2o5 + water in RADM2
	es = 1000.*0.6112*exp(17.67*((wrfphydata[((1)*48*20*48 + (j)*20*48 + (k)*48 + (i))])-273.15)/((wrfphydata[((1)*48*20*48 + (j)*20*48 + (k)*48 + (i))])- 29.65));
	qvs = es / ((wrfphydata[((0)*48*20*48 + (j)*20*48 + (k)*48 + (i))]) - es);
	rh = (wrfmoistdata[((1)*48*20*48 + (j)*20*48 + (k)*48 + (i))]) / qvs;
	rh = MIN ( MAX ( rh, 0.0f), 1.0f);
	*RC_N2O5 = 1.0 / (3.6e4 * exp(-pow((rh/0.28),2.8) + 300.0));

	// convesion from ppmV to molecules/cm3
	conv = 1.0E-6 * dens2con_a * (wrfphydata[((2)*48*20*48 + (j)*20*48 + (k)*48 + (i))]);

	jv[MY_I(0)] = (wrfphdata[((0)*48*20*48 + (j)*20*48 + (k)*48 + (i))]) / 60.0;
	jv[MY_I(1)] = (wrfphdata[((1)*48*20*48 + (j)*20*48 + (k)*48 + (i))]) / 60.0;
	jv[MY_I(2)] = (wrfphdata[((2)*48*20*48 + (j)*20*48 + (k)*48 + (i))]) / 60.0;
	jv[MY_I(3)] = (wrfphdata[((3)*48*20*48 + (j)*20*48 + (k)*48 + (i))]) / 60.0;
	jv[MY_I(4)] = (wrfphdata[((4)*48*20*48 + (j)*20*48 + (k)*48 + (i))]) / 60.0;
	jv[MY_I(5)] = (wrfphdata[((5)*48*20*48 + (j)*20*48 + (k)*48 + (i))]) / 60.0;
	jv[MY_I(6)] = (wrfphdata[((6)*48*20*48 + (j)*20*48 + (k)*48 + (i))]) / 60.0;
	jv[MY_I(7)] = (wrfphdata[((7)*48*20*48 + (j)*20*48 + (k)*48 + (i))]) / 60.0;
	jv[MY_I(8)] = (wrfphdata[((8)*48*20*48 + (j)*20*48 + (k)*48 + (i))]) / 60.0;
	jv[MY_I(9)] = (wrfphdata[((9)*48*20*48 + (j)*20*48 + (k)*48 + (i))]) / 60.0;
	jv[MY_I(10)] = (wrfphdata[((10)*48*20*48 + (j)*20*48 + (k)*48 + (i))]) / 60.0;
	jv[MY_I(11)] = (wrfphdata[((11)*48*20*48 + (j)*20*48 + (k)*48 + (i))]) / 60.0;
	jv[MY_I(12)] = (wrfphdata[((12)*48*20*48 + (j)*20*48 + (k)*48 + (i))]) / 60.0;
	jv[MY_I(13)] = (wrfphdata[((13)*48*20*48 + (j)*20*48 + (k)*48 + (i))]) / 60.0;
	jv[MY_I(14)] = (wrfphdata[((14)*48*20*48 + (j)*20*48 + (k)*48 + (i))]) / 60.0;
	jv[MY_I(15)] = (wrfphdata[((15)*48*20*48 + (j)*20*48 + (k)*48 + (i))]) / 60.0;
	jv[MY_I(16)] = (wrfphdata[((16)*48*20*48 + (j)*20*48 + (k)*48 + (i))]) / 60.0;
	jv[MY_I(17)] = (wrfphdata[((17)*48*20*48 + (j)*20*48 + (k)*48 + (i))]) / 60.0;
	jv[MY_I(18)] = (wrfphdata[((18)*48*20*48 + (j)*20*48 + (k)*48 + (i))]) / 60.0;
	jv[MY_I(19)] = (wrfphdata[((19)*48*20*48 + (j)*20*48 + (k)*48 + (i))]) / 60.0;
	jv[MY_I(20)] = (wrfphdata[((20)*48*20*48 + (j)*20*48 + (k)*48 + (i))]) / 60.0;
	jv[MY_I(21)] = (wrfphdata[((21)*48*20*48 + (j)*20*48 + (k)*48 + (i))]) / 60.0;
	jv[MY_I(22)] = (wrfphdata[((22)*48*20*48 + (j)*20*48 + (k)*48 + (i))]) / 60.0;
	jv[MY_I(23)] = (wrfphdata[((23)*48*20*48 + (j)*20*48 + (k)*48 + (i))]) / 60.0;
	jv[MY_I(24)] = (wrfphdata[((24)*48*20*48 + (j)*20*48 + (k)*48 + (i))]) / 60.0;
	jv[MY_I(25)] = (wrfphdata[((25)*48*20*48 + (j)*20*48 + (k)*48 + (i))]) / 60.0;
	jv[MY_I(26)] = (wrfphdata[((26)*48*20*48 + (j)*20*48 + (k)*48 + (i))]) / 60.0;
	jv[MY_I(27)] = (wrfphdata[((27)*48*20*48 + (j)*20*48 + (k)*48 + (i))]) / 60.0;
	
	// Shuffle and convert concentration data
	VAR[MY_I(45)] = conv * MAX((wrfchemdata[((5)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(13)] = conv * MAX((wrfchemdata[((7)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(57)] = conv * MAX((wrfchemdata[((4)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(58)] = conv * MAX((wrfchemdata[((3)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(53)] = conv * MAX((wrfchemdata[((17)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(14)] = conv * MAX((wrfchemdata[((16)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(12)] = conv * MAX((wrfchemdata[((31)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(24)] = conv * MAX((wrfchemdata[((6)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(19)] = conv * MAX((wrfchemdata[((32)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(4)]  = conv * MAX((wrfchemdata[((1)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(0)] = conv * MAX((wrfchemdata[((2)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(25)]  = conv * MAX((wrfchemdata[((23)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(5)] = conv * MAX((wrfchemdata[((22)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(15)] = conv * MAX((wrfchemdata[((19)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(7)] = conv * MAX((wrfchemdata[((20)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(8)] = conv * MAX((wrfchemdata[((21)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(23)] = conv * MAX((wrfchemdata[((24)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(27)] = conv * MAX((wrfchemdata[((25)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(28)] = conv * MAX((wrfchemdata[((26)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(26)] = conv * MAX((wrfchemdata[((39)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(9)] = conv * MAX((wrfchemdata[((27)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(10)] = conv * MAX((wrfchemdata[((28)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(21)] = conv * MAX((wrfchemdata[((38)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(42)] = conv * MAX((wrfchemdata[((9)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(47)] = conv * MAX((wrfchemdata[((8)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(56)] = conv * MAX((wrfraddata[((15)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(32)] = conv * MAX((wrfchemdata[((33)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(30)] = conv * MAX((wrfchemdata[((34)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(33)] = conv * MAX((wrfchemdata[((35)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(51)] = conv * MAX((wrfraddata[((18)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(29)] = conv * MAX((wrfchemdata[((36)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(46)] = conv * MAX((wrfchemdata[((37)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(22)] = conv * MAX((wrfchemdata[((18)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(11)] = conv * MAX((wrfchemdata[((30)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(20)] = conv * MAX((wrfchemdata[((10)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(41)] = conv * MAX((wrfchemdata[((11)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(17)] = conv * MAX((wrfchemdata[((12)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(1)] = conv * MAX((wrfchemdata[((13)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(2)] = conv * MAX((wrfchemdata[((14)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(52)] = conv * MAX((wrfchemdata[((42)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(50)] = conv * MAX((wrfchemdata[((43)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(18)] = conv * MAX((wrfraddata[((16)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(6)] = conv * MAX((wrfraddata[((19)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(55)] = conv * MAX((wrfraddata[((14)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(40)] = conv * MAX((wrfraddata[((8)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(43)] = conv * MAX((wrfraddata[((9)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(36)] = conv * MAX((wrfraddata[((4)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(48)] = conv * MAX((wrfraddata[((5)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(34)] = conv * MAX((wrfraddata[((10)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(35)] = conv * MAX((wrfraddata[((11)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(54)] = conv * MAX((wrfchemdata[((29)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(44)] = conv * MAX((wrfraddata[((17)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(49)] = conv * MAX((wrfraddata[((24)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(37)] = conv * MAX((wrfraddata[((27)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(38)] = conv * MAX((wrfraddata[((23)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(31)] = conv * MAX((wrfraddata[((25)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(16)] = conv * MAX((wrfchemdata[((41)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(3)] = conv * MAX((wrfchemdata[((40)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
	VAR[MY_I(39)] = conv * MAX((wrfchemdata[((26)*48*20*48 + (j)*20*48 + (k)*48 + (i))]),0.0);
}

void convertWRFChemData() {
	dim3 gridDim, blockDim;

	gridDim.y  = 39;
	gridDim.x  = 19;
	blockDim.x = 39;

	cudaStartTimer(&timingconvert, 0, 0);

	for(device = startDevice; device < endDevice; device++) {
		cudaSetDevice(device);
		
		dev_convert<<<gridDim, blockDim>>>(
			DataDevice[device].d_VAR, DataDevice[device].d_FIX, 
			DataDevice[device].d_TEMP, DataDevice[device].d_CFACTOR, 
			DataDevice[device].d_RC_N2O5, DataDevice[device].d_JV, 
			DataDevice[device].d_refwrfphdata, DataDevice[device].d_refwrfchemdata, 
			DataDevice[device].d_refwrfraddata,
			DataDevice[device].d_refwrfphydata, 
			DataDevice[device].d_refwrfmoistdata);
	}
	
	for(device = startDevice; device < endDevice; device++) {
		cudaSetDevice(device);
	
		cudaDeviceSynchronize();
	}

	cudaStopTimer(&timingconvert, 0, 0);

	printf("Converting WRF-Chem device data\n");
	printf("Timing = %f ms\n\n", timingconvert.elapsed);
}

void recoverDeviceMemory() {
	for(device = startDevice; device < endDevice; device++) {
		cudaSetDevice(device);

		cudaFree(DataDevice[device].d_wrf_chem);
    		cudaFree(DataDevice[device].d_wrf_moist);
    		cudaFree(DataDevice[device].d_wrf_vdrog3);
    		cudaFree(DataDevice[device].d_wrf_phy);
    		cudaFree(DataDevice[device].d_wrf_rad);
    		cudaFree(DataDevice[device].d_wrf_ph);

		cudaFree(DataDevice[device].d_refwrfchemdata);
    		cudaFree(DataDevice[device].d_refwrfmoistdata);
    		cudaFree(DataDevice[device].d_refwrfvdrog3data);
    		cudaFree(DataDevice[device].d_refwrfphydata);
    		cudaFree(DataDevice[device].d_refwrfraddata);
    		cudaFree(DataDevice[device].d_refwrfphdata);
	}

	for(device = startDevice; device < endDevice; device++) {
		cudaSetDevice(device);
	
		cudaDeviceSynchronize();
	}
}

__device__ double ARR2(double A0, double B0, double TEMP) {
	return A0 * exp( -B0/TEMP);
}

// Troe reactions (as in Stockwell et al, 1997)
__device__ double TROE(double k0_300K, double n, double kinf_300K, double m, double temp, double cair) {
	double zt_help, k0_T, kinf_T, k_ratio;
	
	zt_help = 300.0/temp;
	k0_T    = k0_300K * pow(zt_help, n) * cair; // k_0   at current T
	kinf_T  = kinf_300K * pow(zt_help, m);      // k_inf at current T
	k_ratio = k0_T/kinf_T;

	return k0_T / (1.0 + k_ratio) * pow(0.6, 1.0/(1.0+pow(log10(k_ratio), 2)));
}

// Troe equilibrium reactions (as in Stockwell et al, 1997)
__device__ double TROEE(double A, double B, double k0_300K, double n, double kinf_300K, double m, double temp, double cair) {
	double zt_help, k0_T, kinf_T, k_ratio, troe;

	zt_help = 300.0/temp;
	k0_T    = k0_300K   * pow(zt_help,n) * cair; // k_0   at current T
	kinf_T  = kinf_300K * pow(zt_help,m);        // k_inf at current T
	k_ratio = k0_T/kinf_T;
	troe    = k0_T / (1.0 + k_ratio) * pow(0.6, 1.0/(1.0+pow(log10(k_ratio), 2)));

	return A * exp( -B / temp) * troe;
}

// k=T^2 C exp (-D/T) reactions
__device__ double THERMAL_T2(double c, double d, double TEMP) {
	return pow(TEMP, (double)2.0) * c * exp(-d / TEMP);
}

__device__ double k46(double C_M, double TEMP) {
	double k0, k2, k3;

	k0=7.2e-15 * exp(785.0/TEMP);
	k2=4.1e-16 * exp(1440.0/TEMP);
	k3=1.9e-33 * exp(725.0/TEMP) * C_M;

	return k0+k3/(1+k3/k2);
}

__global__ void dev_Update_RCONST(
	double * RCONST, double * jv, double * FIX, double * RC_N2O5, double * T) {
	int i = threadIdx.x;
	int j = blockIdx.y;
	int k = blockIdx.x;
	int p = j*gridDim.x*blockDim.x + k*blockDim.x + i;

	double C_M = FIX[MY_I(indf_M)];
	double rc_n2o5 = RC_N2O5[p];
	double TEMP = T[p];
	
	RCONST[MY_I(0)] = (jv[MY_I(2)]);
	RCONST[MY_I(1)] = (jv[MY_I(0)]);
	RCONST[MY_I(2)] = (jv[MY_I(1)]);
	RCONST[MY_I(3)] = (jv[MY_I(5)]);
	RCONST[MY_I(4)] = (jv[MY_I(6)]);
	RCONST[MY_I(5)] = (jv[MY_I(7)]);
	RCONST[MY_I(6)] = (jv[MY_I(3)]);
	RCONST[MY_I(7)] = (jv[MY_I(4)]);
	RCONST[MY_I(8)] = (jv[MY_I(8)]);
	RCONST[MY_I(9)] = (jv[MY_I(10)]);
	RCONST[MY_I(10)] = (jv[MY_I(9)]);
	RCONST[MY_I(11)] = (jv[MY_I(11)]);
	RCONST[MY_I(12)] = (jv[MY_I(17)]);
	RCONST[MY_I(13)] = (jv[MY_I(12)]);
	RCONST[MY_I(14)] = (jv[MY_I(18)]);
	RCONST[MY_I(15)] = (jv[MY_I(13)]);
	RCONST[MY_I(16)] = (jv[MY_I(14)]);
	RCONST[MY_I(17)] = (jv[MY_I(20)]);
	RCONST[MY_I(18)] = (jv[MY_I(15)]);
	RCONST[MY_I(19)] = (jv[MY_I(16)]);
	RCONST[MY_I(20)] = (jv[MY_I(19)]);
	RCONST[MY_I(21)] = (.20946e0*(C_M*6.00e-34*pow(TEMP/300.0,-2.3)));
	RCONST[MY_I(22)] = (ARR2(6.5e-12, -120.0, TEMP));
	RCONST[MY_I(23)] = (.78084*ARR2(1.8e-11, -110.0, TEMP)+.20946e0*ARR2(3.2e-11,-70.0, TEMP));
	RCONST[MY_I(24)] = (2.2e-10);
	RCONST[MY_I(25)] = (ARR2(2.0e-12, 1400.0, TEMP));
	RCONST[MY_I(26)] = (ARR2(1.6e-12, 940.0, TEMP));
	RCONST[MY_I(27)] = (ARR2(1.1e-14, 500.0, TEMP));
	RCONST[MY_I(28)] = (ARR2(3.7e-12, -240.0, TEMP));
	RCONST[MY_I(29)] = (TROE(1.80e-31,3.2,4.70e-12,1.4,TEMP,C_M));
	RCONST[MY_I(30)] = (TROEE(4.76e26,10900.0,1.80e-31,3.2,4.70e-12,1.4,TEMP,C_M));
	RCONST[MY_I(31)] = ((2.2e-13*exp(600./TEMP)+1.9e-33*C_M*exp(980./TEMP)));
	RCONST[MY_I(32)] = ((3.08e-34*exp(2800./TEMP)+2.66e-54*C_M*exp(3180./TEMP)));
	RCONST[MY_I(33)] = (ARR2(3.3e-12, 200.0, TEMP));
	RCONST[MY_I(34)] = (TROE(7.00e-31,2.6,1.50e-11,0.5,TEMP,C_M));
	RCONST[MY_I(35)] = (.20946e0*ARR2(3.3e-39, -530.0, TEMP));
	RCONST[MY_I(36)] = (ARR2(1.4e-13, 2500.0, TEMP));
	RCONST[MY_I(37)] = (ARR2(1.7e-11, -150.0, TEMP));
	RCONST[MY_I(38)] = (ARR2(2.5e-14, 1230.0, TEMP));
	RCONST[MY_I(39)] = (2.5e-12);
	RCONST[MY_I(40)] = (TROE(2.20e-30,4.3,1.50e-12,0.5,TEMP,C_M));
	RCONST[MY_I(41)] = (TROEE(9.09e26,11200.0,2.20e-30,4.3,1.50e-12,0.5,TEMP,C_M));
	RCONST[MY_I(42)] = (rc_n2o5);
	RCONST[MY_I(43)] = (TROE(2.60e-30,3.2,2.40e-11,1.3,TEMP,C_M));
	RCONST[MY_I(44)] = (k46(C_M, TEMP));
	RCONST[MY_I(45)] = (ARR2(1.3e-12, -380.0, TEMP));
	RCONST[MY_I(46)] = (ARR2(4.6e-11, -230.0, TEMP));
	RCONST[MY_I(47)] = (TROE(3.00e-31,3.3,1.50e-12,0.0,TEMP,C_M));
	RCONST[MY_I(48)] = ((1.5e-13*(1.+2.439e-20*C_M)));
	RCONST[MY_I(49)] = (THERMAL_T2(6.95e-18, 1280.0, TEMP));
	RCONST[MY_I(50)] = (THERMAL_T2(1.37e-17, 444.0, TEMP));
	RCONST[MY_I(51)] = (ARR2(1.59e-11, 540.0, TEMP));
	RCONST[MY_I(52)] = (ARR2(1.73e-11, 380.0, TEMP));
	RCONST[MY_I(53)] = (ARR2(3.64e-11, 380.0, TEMP));
	RCONST[MY_I(54)] = (ARR2(2.15e-12, -411.0, TEMP));
	RCONST[MY_I(55)] = (ARR2(5.32e-12, -504.0, TEMP));
	RCONST[MY_I(56)] = (ARR2(1.07e-11, -549.0, TEMP));
	RCONST[MY_I(57)] = (ARR2(2.1e-12, -322.0, TEMP));
	RCONST[MY_I(58)] = (ARR2(1.89e-11, -116.0, TEMP));
	RCONST[MY_I(59)] = (4.0e-11);
	RCONST[MY_I(60)] = (9.0e-12);
	RCONST[MY_I(61)] = (ARR2(6.87e-12, -256.0, TEMP));
	RCONST[MY_I(62)] = (ARR2(1.2e-11, 745.0, TEMP));
	RCONST[MY_I(63)] = (1.15e-11);
	RCONST[MY_I(64)] = (1.7e-11);
	RCONST[MY_I(65)] = (2.8e-11);
	RCONST[MY_I(66)] = (1.0e-11);
	RCONST[MY_I(67)] = (1.0e-11);
	RCONST[MY_I(68)] = (1.0e-11);
	RCONST[MY_I(69)] = (THERMAL_T2(6.85e-18, 444.0, TEMP));
	RCONST[MY_I(70)] = (ARR2(1.55e-11, 540.0, TEMP));
	RCONST[MY_I(71)] = (ARR2(2.55e-11, -409.0, TEMP));
	RCONST[MY_I(72)] = (ARR2(2.8e-12, -181.0, TEMP));
	RCONST[MY_I(73)] = (ARR2(1.95e+16,13543.0, TEMP));
	RCONST[MY_I(74)] = (4.7e-12);
	RCONST[MY_I(75)] = (ARR2(1.95e+16,13543.0, TEMP));
	RCONST[MY_I(76)] = (ARR2(4.2e-12, -180.0, TEMP));
	RCONST[MY_I(77)] = (ARR2(4.2e-12, -180.0, TEMP));
	RCONST[MY_I(78)] = (ARR2(4.2e-12, -180.0, TEMP));
	RCONST[MY_I(79)] = (ARR2(4.2e-12, -180.0, TEMP));
	RCONST[MY_I(80)] = (ARR2(4.2e-12, -180.0, TEMP));
	RCONST[MY_I(81)] = (ARR2(4.2e-12, -180.0, TEMP));
	RCONST[MY_I(82)] = (ARR2(4.2e-12, -180.0, TEMP));
	RCONST[MY_I(83)] = (ARR2(4.2e-12, -180.0, TEMP));
	RCONST[MY_I(84)] = (ARR2(4.2e-12, -180.0, TEMP));
	RCONST[MY_I(85)] = (ARR2(4.2e-12, -180.0, TEMP));
	RCONST[MY_I(86)] = (ARR2(4.2e-12, -180.0, TEMP));
	RCONST[MY_I(87)] = (ARR2(4.2e-12, -180.0, TEMP));
	RCONST[MY_I(88)] = (ARR2(4.2e-12, -180.0, TEMP));
	RCONST[MY_I(89)] = (ARR2(4.2e-12, -180.0, TEMP));
	RCONST[MY_I(90)] = (ARR2(6.0e-13, 2058.0, TEMP));
	RCONST[MY_I(91)] = (ARR2(1.4e-12, 1900.0, TEMP));
	RCONST[MY_I(92)] = (ARR2(6.0e-13, 2058.0, TEMP));
	RCONST[MY_I(93)] = (ARR2(1.4e-12, 1900.0, TEMP));
	RCONST[MY_I(94)] = (ARR2(1.4e-12, 1900.0, TEMP));
	RCONST[MY_I(95)] = (2.2e-11);
	RCONST[MY_I(96)] = (ARR2(2.0e-12, 2923.0, TEMP));
	RCONST[MY_I(97)] = (ARR2(1.0e-11, 1895.0, TEMP));
	RCONST[MY_I(98)] = (ARR2(3.23e-11, 975.0, TEMP));
	RCONST[MY_I(99)] = (5.81e-13);
	RCONST[MY_I(100)] = (ARR2(1.2e-14, 2633.0, TEMP));
	RCONST[MY_I(101)] = (ARR2(1.32e-14, 2105.0, TEMP));
	RCONST[MY_I(102)] = (ARR2(7.29e-15, 1136.0, TEMP));
	RCONST[MY_I(103)] = (ARR2(1.23e-14, 2013.0, TEMP));
	RCONST[MY_I(104)] = (ARR2(7.7e-14, -1300.0, TEMP));
	RCONST[MY_I(105)] = (ARR2(7.7e-14, -1300.0, TEMP));
	RCONST[MY_I(106)] = (ARR2(7.7e-14, -1300.0, TEMP));
	RCONST[MY_I(107)] = (ARR2(7.7e-14, -1300.0, TEMP));
	RCONST[MY_I(108)] = (ARR2(7.7e-14, -1300.0, TEMP));
	RCONST[MY_I(109)] = (ARR2(7.7e-14, -1300.0, TEMP));
	RCONST[MY_I(110)] = (ARR2(7.7e-14, -1300.0, TEMP));
	RCONST[MY_I(111)] = (ARR2(7.7e-14, -1300.0, TEMP));
	RCONST[MY_I(112)] = (ARR2(7.7e-14, -1300.0, TEMP));
	RCONST[MY_I(113)] = (ARR2(7.7e-14, -1300.0, TEMP));
	RCONST[MY_I(114)] = (ARR2(7.7e-14, -1300.0, TEMP));
	RCONST[MY_I(115)] = (ARR2(7.7e-14, -1300.0, TEMP));
	RCONST[MY_I(116)] = (ARR2(7.7e-14, -1300.0, TEMP));
	RCONST[MY_I(117)] = (ARR2(7.7e-14, -1300.0, TEMP));
	RCONST[MY_I(118)] = (ARR2(1.9e-13, -220.0, TEMP));
	RCONST[MY_I(119)] = (ARR2(1.4e-13, -220.0, TEMP));
	RCONST[MY_I(120)] = (ARR2(4.2e-14, -220.0, TEMP));
	RCONST[MY_I(121)] = (ARR2(3.4e-14, -220.0, TEMP));
	RCONST[MY_I(122)] = (ARR2(2.9e-14, -220.0, TEMP));
	RCONST[MY_I(123)] = (ARR2(1.4e-13, -220.0, TEMP));
	RCONST[MY_I(124)] = (ARR2(1.4e-13, -220.0, TEMP));
	RCONST[MY_I(125)] = (ARR2(1.7e-14, -220.0, TEMP));
	RCONST[MY_I(126)] = (ARR2(1.7e-14, -220.0, TEMP));
	RCONST[MY_I(127)] = (ARR2(9.6e-13, -220.0, TEMP));
	RCONST[MY_I(128)] = (ARR2(1.7e-14, -220.0, TEMP));
	RCONST[MY_I(129)] = (ARR2(1.7e-14, -220.0, TEMP));
	RCONST[MY_I(130)] = (ARR2(9.6e-13, -220.0, TEMP));
	RCONST[MY_I(131)] = (ARR2(3.4e-13, -220.0, TEMP));
	RCONST[MY_I(132)] = (ARR2(1.0e-13, -220.0, TEMP));
	RCONST[MY_I(133)] = (ARR2(8.4e-14, -220.0, TEMP));
	RCONST[MY_I(134)] = (ARR2(7.2e-14, -220.0, TEMP));
	RCONST[MY_I(135)] = (ARR2(3.4e-13, -220.0, TEMP));
	RCONST[MY_I(136)] = (ARR2(3.4e-13, -220.0, TEMP));
	RCONST[MY_I(137)] = (ARR2(4.2e-14, -220.0, TEMP));
	RCONST[MY_I(138)] = (ARR2(4.2e-14, -220.0, TEMP));
	RCONST[MY_I(139)] = (ARR2(1.19e-12, -220.0, TEMP));
	RCONST[MY_I(140)] = (ARR2(4.2e-14, -220.0, TEMP));
	RCONST[MY_I(141)] = (ARR2(4.2e-14, -220.0, TEMP));
	RCONST[MY_I(142)] = (ARR2(1.19e-12, -220.0, TEMP));
	RCONST[MY_I(143)] = (ARR2(7.7e-14, -1300.0, TEMP));
	RCONST[MY_I(144)] = (ARR2(1.7e-14, -220.0, TEMP));
	RCONST[MY_I(145)] = (ARR2(4.2e-14, -220.0, TEMP));
	RCONST[MY_I(146)] = (ARR2(3.6e-16, -220.0, TEMP));
	RCONST[MY_I(147)] = (ARR2(4.2e-12, -180.0, TEMP));
	RCONST[MY_I(148)] = (ARR2(4.2e-12, -180.0, TEMP));
	RCONST[MY_I(149)] = (ARR2(7.7e-14, -1300.0, TEMP));
	RCONST[MY_I(150)] = (ARR2(1.7e-14, -220.0, TEMP));
	RCONST[MY_I(151)] = (ARR2(4.2e-14, -220.0, TEMP));
	RCONST[MY_I(152)] = (ARR2(3.6e-16, -220.0, TEMP));
	RCONST[MY_I(153)] = (ARR2(1.7e-14, -220.0, TEMP));
	RCONST[MY_I(154)] = (ARR2(4.2e-14, -220.0, TEMP));
	RCONST[MY_I(155)] = (ARR2(3.6e-16, -220.0, TEMP));
}

void updatingCoefficientWRFChemData() {
	dim3 gridDim, blockDim;
	gridDim.y = 39;
	gridDim.x = 19;
	blockDim.x = 39;

	cudaStartTimer(&timingupdaterconst, 0, 0);

	for(device = startDevice; device < endDevice; device++) {
		cudaSetDevice(device);

		dev_Update_RCONST<<<gridDim, blockDim>>>(
		DataDevice[device].d_RCONST, DataDevice[device].d_JV,
		DataDevice[device].d_FIX, DataDevice[device].d_RC_N2O5, DataDevice[device].d_TEMP);
	}

	for(device = startDevice; device < endDevice; device++) {
		cudaSetDevice(device);
	
		cudaDeviceSynchronize();
	}

	cudaStopTimer(&timingupdaterconst, 0, 0);

	printf("Updating and calculating rate coefficients\n");
	printf("Timing = %f ms\n\n", timingupdaterconst.elapsed);
}

__host__ double WLAMCH(char C) {
	int i;
	double Suma;
	static double Eps;
	static char First = 1;

	if (First) {
		First = 0;
		Eps = pow(0.5, 16);
		for (i = 17; i <= 80; i++) {
			Eps = Eps * 0.5;
			Suma = 1.0 + Eps;
			if (Suma <= 1.0)
				break;
		}
		if (i == 80) {
			//printf("\nERROR IN WLAMCH. Very small EPS = %g\n", Eps);
			return (double) 2.2e-16;
		}
		Eps *= 2.0;
		i--;
	}

	return Eps;
}

// Handles all error messages and returns IERR = error Code
int ros_ErrorMsg(int Code, double T, double H) {
   	printf("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
   	printf("\nForced exit from Rosenbrock due to the following error:\n");

   	switch (Code) {
   		case -1:
      			printf("--> Improper value for maximal no of steps"); break;
   		case -2:
      			printf("--> Selected Rosenbrock method not implemented"); break;
   		case -3:
      			printf("--> Hmin/Hmax/Hstart must be positive"); break;
   		case -4:
      			printf("--> FacMin/FacMax/FacRej must be positive"); break;
   		case -5:
     			printf("--> Improper tolerance values"); break;
   		case -6:
      			printf("--> No of steps exceeds maximum bound"); break;
   		case -7:
      			printf("--> Step size too small (T + H/10 = T) or H < Roundoff"); break;
   		case -8:
      			printf("--> Matrix is repeatedly singular"); break;
   		default:
      			printf("Unknown Error code: %d ",Code);
   	}

   	printf("\n   Time = %15.7e,  H = %15.7e",T,H);
	printf("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");

   	return Code;
}

// AN L-STABLE METHOD, 3 stages, order 3, 2 function evaluations
void Ros3( int *ros_S, double ros_A[], double ros_C[],
           double ros_M[], double ros_E[],
	   double ros_Alpha[], double ros_Gamma[],
	   char ros_NewF[], double *ros_ELO, char* ros_Name ) {

	// Name of the method
   	strcpy(ros_Name, "ROS-3");

  	// Number of stages
   	*ros_S = 3;

  	// The coefficient matrices A and C are strictly lower triangular.
    	// The lower triangular (subdiagonal) elements are stored in row-wise order:
    	// A(2,1) = ros_A[0], A(3,1)=ros_A[1], A(3,2)=ros_A[2], etc.
    	// The general mapping formula is:
        // A_{i,j} = ros_A[ (i-1)*(i-2)/2 + j -1 ]
   	ros_A[0]= (double)1.0;
   	ros_A[1]= (double)1.0;
   	ros_A[2]= (double)0.0;

  	//C_{i,j} = ros_C[ (i-1)*(i-2)/2 + j -1]
   	ros_C[0] = (double)(-1.0156171083877702091975600115545);
   	ros_C[1] = (double)4.0759956452537699824805835358067;
   	ros_C[2] = (double)9.2076794298330791242156818474003;

  	// does the stage i require a new function evaluation (ros_NewF(i)=TRUE)
    	// or does it re-use the function evaluation from stage i-1 (ros_NewF(i)=FALSE)
   	ros_NewF[0] = 1;
   	ros_NewF[1] = 1;
   	ros_NewF[2] = 0;

  	// M_i = Coefficients for new step solution
   	ros_M[0] = (double)1.0;
   	ros_M[1] = (double)6.1697947043828245592553615689730;
   	ros_M[2] = (double)(-0.4277225654321857332623837380651);

  	// E_i = Coefficients for error estimator
   	ros_E[0] = (double)0.5;
   	ros_E[1] = (double)(-2.9079558716805469821718236208017);
   	ros_E[2] = (double)0.2235406989781156962736090927619;

  	// ros_ELO = estimator of local order - the minimum between the
	// !    main and the embedded scheme orders plus 1
   	*ros_ELO = (double)3.0;

  	// Y_stage_i ~ Y( T + H*Alpha_i )
   	ros_Alpha[0]= (double)0.0;
   	ros_Alpha[1]= (double)0.43586652150845899941601945119356;
   	ros_Alpha[2]= (double)0.43586652150845899941601945119356;

  	// Gamma_i = \sum_j  gamma_{i,j}
   	ros_Gamma[0]= (double)0.43586652150845899941601945119356;
   	ros_Gamma[1]= (double)0.24291996454816804366592249683314;
   	ros_Gamma[2]= (double)2.1851380027664058511513169485832;
}

__global__ void dev_setVectorReal(int length, double value, double * d_V) {
	int i = threadIdx.x;
	int j = blockIdx.y;
	int k = blockIdx.x;
	int p = j*gridDim.x*blockDim.x + k*blockDim.x + i;
        if ( p >= length ) return ;
        d_V[p] = value ;
}

__host__ void setVectorReal(double value, double * d_V, int N) {
        dim3 gridDim, blockDim;
        // One-thread-per-cell decomposition
        // 512 threads per block
        blockDim.x = 512;
        gridDim.x = N / blockDim.x + (N % blockDim.x > 0);

        dev_setVectorReal <<<gridDim, blockDim>>> ( N, value, d_V ) ;

}

__global__ void dev_Fun(int length, double * V, double * F, double * RCT, double * Vdot) {
	
	int p = blockIdx.x*blockDim.x + threadIdx.x;

	if(p >= length) return;

        //if( vecMask[p] == 0 || vecMask2[p] == 0 ) return;

	// Adjust pointers to select desired cell
	V    = &V[p*NVAR];
	F    = &F[p*NFIX];
	RCT  = &RCT[p*NREACT];
	Vdot = &Vdot[p*NVAR];

	double A[156];

	#define MY_I_A(i) MY_I(i)
	#define MY_I(i) (i)
	
	A[MY_I_A(0)] = RCT[MY_I(0)] * V[MY_I(58)];
	A[MY_I_A(1)] = RCT[MY_I(1)] * V[MY_I(45)];
	A[MY_I_A(2)] = RCT[MY_I(2)] * V[MY_I(45)];
	A[MY_I_A(3)] = RCT[MY_I(3)] * V[MY_I(12)];
	A[MY_I_A(4)] = RCT[MY_I(4)] * V[MY_I(24)];
	A[MY_I_A(5)] = RCT[MY_I(5)] * V[MY_I(19)];
	A[MY_I_A(6)] = RCT[MY_I(6)] * V[MY_I(53)];
	A[MY_I_A(7)] = RCT[MY_I(7)] * V[MY_I(53)];
	A[MY_I_A(8)] = RCT[MY_I(8)] * V[MY_I(13)];
	A[MY_I_A(9)] = RCT[MY_I(9)] * V[MY_I(42)];
	A[MY_I_A(10)] = RCT[MY_I(10)] * V[MY_I(42)];
	A[MY_I_A(11)] = RCT[MY_I(11)] * V[MY_I(47)];
	A[MY_I_A(12)] = RCT[MY_I(12)] * V[MY_I(20)];
	A[MY_I_A(13)] = RCT[MY_I(13)] * V[MY_I(41)];
	A[MY_I_A(14)] = RCT[MY_I(14)] * V[MY_I(17)];
	A[MY_I_A(15)] = RCT[MY_I(15)] * V[MY_I(32)];
	A[MY_I_A(16)] = RCT[MY_I(16)] * V[MY_I(30)];
	A[MY_I_A(17)] = RCT[MY_I(17)] * V[MY_I(30)];
	A[MY_I_A(18)] = RCT[MY_I(18)] * V[MY_I(33)];
	A[MY_I_A(19)] = RCT[MY_I(19)] * V[MY_I(29)];
	A[MY_I_A(20)] = RCT[MY_I(20)] * V[MY_I(46)];
	A[MY_I_A(21)] = RCT[MY_I(21)] * V[MY_I(18)] * F[MY_I(1)];
	A[MY_I_A(22)] = RCT[MY_I(22)] * V[MY_I(18)] * V[MY_I(58)];
	A[MY_I_A(23)] = RCT[MY_I(23)] * V[MY_I(6)] * F[MY_I(1)];
	A[MY_I_A(24)] = RCT[MY_I(24)] * V[MY_I(6)] * F[MY_I(0)];
	A[MY_I_A(25)] = RCT[MY_I(25)] * V[MY_I(45)] * V[MY_I(57)];
	A[MY_I_A(26)] = RCT[MY_I(26)] * V[MY_I(45)] * V[MY_I(52)];
	A[MY_I_A(27)] = RCT[MY_I(27)] * V[MY_I(45)] * V[MY_I(50)];
	A[MY_I_A(28)] = RCT[MY_I(28)] * V[MY_I(50)] * V[MY_I(57)];
	A[MY_I_A(29)] = RCT[MY_I(29)] * V[MY_I(50)] * V[MY_I(58)];
	A[MY_I_A(30)] = RCT[MY_I(30)] * V[MY_I(19)];
	A[MY_I_A(31)] = RCT[MY_I(31)] * V[MY_I(50)] * V[MY_I(50)];
	A[MY_I_A(32)] = RCT[MY_I(32)] * V[MY_I(50)] * V[MY_I(50)] * F[MY_I(0)];
	A[MY_I_A(33)] = RCT[MY_I(33)] * V[MY_I(13)] * V[MY_I(52)];
	A[MY_I_A(34)] = RCT[MY_I(34)] * V[MY_I(52)] * V[MY_I(57)];
	A[MY_I_A(35)] = RCT[MY_I(35)] * V[MY_I(57)] * V[MY_I(57)] * F[MY_I(1)];
	A[MY_I_A(36)] = RCT[MY_I(36)] * V[MY_I(45)] * V[MY_I(58)];
	A[MY_I_A(37)] = RCT[MY_I(37)] * V[MY_I(53)] * V[MY_I(57)];
	A[MY_I_A(38)] = RCT[MY_I(38)] * V[MY_I(53)] * V[MY_I(58)];
	A[MY_I_A(39)] = RCT[MY_I(39)] * V[MY_I(50)] * V[MY_I(53)];
	A[MY_I_A(40)] = RCT[MY_I(40)] * V[MY_I(53)] * V[MY_I(58)];
	A[MY_I_A(41)] = RCT[MY_I(41)] * V[MY_I(14)];
	A[MY_I_A(42)] = RCT[MY_I(42)] * V[MY_I(14)];
	A[MY_I_A(43)] = RCT[MY_I(43)] * V[MY_I(52)] * V[MY_I(58)];
	A[MY_I_A(44)] = RCT[MY_I(44)] * V[MY_I(24)] * V[MY_I(52)];
	A[MY_I_A(45)] = RCT[MY_I(45)] * V[MY_I(19)] * V[MY_I(52)];
	A[MY_I_A(46)] = RCT[MY_I(46)] * V[MY_I(50)] * V[MY_I(52)];
	A[MY_I_A(47)] = RCT[MY_I(47)] * V[MY_I(4)] * V[MY_I(52)];
	A[MY_I_A(48)] = RCT[MY_I(48)] * V[MY_I(25)] * V[MY_I(52)];
	A[MY_I_A(49)] = RCT[MY_I(49)] * V[MY_I(16)] * V[MY_I(52)];
	A[MY_I_A(50)] = RCT[MY_I(50)] * V[MY_I(5)] * V[MY_I(52)];
	A[MY_I_A(51)] = RCT[MY_I(51)] * V[MY_I(15)] * V[MY_I(52)];
	A[MY_I_A(52)] = RCT[MY_I(52)] * V[MY_I(7)] * V[MY_I(52)];
	A[MY_I_A(53)] = RCT[MY_I(53)] * V[MY_I(8)] * V[MY_I(52)];
	A[MY_I_A(54)] = RCT[MY_I(54)] * V[MY_I(23)] * V[MY_I(52)];
	A[MY_I_A(55)] = RCT[MY_I(55)] * V[MY_I(27)] * V[MY_I(52)];
	A[MY_I_A(56)] = RCT[MY_I(56)] * V[MY_I(28)] * V[MY_I(52)];
	A[MY_I_A(57)] = RCT[MY_I(57)] * V[MY_I(9)] * V[MY_I(52)];
	A[MY_I_A(58)] = RCT[MY_I(58)] * V[MY_I(10)] * V[MY_I(52)];
	A[MY_I_A(59)] = RCT[MY_I(59)] * V[MY_I(21)] * V[MY_I(52)];
	A[MY_I_A(60)] = RCT[MY_I(60)] * V[MY_I(42)] * V[MY_I(52)];
	A[MY_I_A(61)] = RCT[MY_I(61)] * V[MY_I(47)] * V[MY_I(52)];
	A[MY_I_A(62)] = RCT[MY_I(62)] * V[MY_I(32)] * V[MY_I(52)];
	A[MY_I_A(63)] = RCT[MY_I(63)] * V[MY_I(30)] * V[MY_I(52)];
	A[MY_I_A(64)] = RCT[MY_I(64)] * V[MY_I(33)] * V[MY_I(52)];
	A[MY_I_A(65)] = RCT[MY_I(65)] * V[MY_I(29)] * V[MY_I(52)];
	A[MY_I_A(66)] = RCT[MY_I(66)] * V[MY_I(20)] * V[MY_I(52)];
	A[MY_I_A(67)] = RCT[MY_I(67)] * V[MY_I(41)] * V[MY_I(52)];
	A[MY_I_A(68)] = RCT[MY_I(68)] * V[MY_I(17)] * V[MY_I(52)];
	A[MY_I_A(69)] = RCT[MY_I(69)] * V[MY_I(22)] * V[MY_I(52)];
	A[MY_I_A(70)] = RCT[MY_I(70)] * V[MY_I(46)] * V[MY_I(52)];
	A[MY_I_A(71)] = RCT[MY_I(71)] * V[MY_I(26)] * V[MY_I(52)];
	A[MY_I_A(72)] = RCT[MY_I(72)] * V[MY_I(54)] * V[MY_I(58)];
	A[MY_I_A(73)] = RCT[MY_I(73)] * V[MY_I(22)];
	A[MY_I_A(74)] = RCT[MY_I(74)] * V[MY_I(44)] * V[MY_I(58)];
	A[MY_I_A(75)] = RCT[MY_I(75)] * V[MY_I(11)];
	A[MY_I_A(76)] = RCT[MY_I(76)] * V[MY_I(51)] * V[MY_I(57)];
	A[MY_I_A(77)] = RCT[MY_I(77)] * V[MY_I(55)] * V[MY_I(57)];
	A[MY_I_A(78)] = RCT[MY_I(78)] * V[MY_I(40)] * V[MY_I(57)];
	A[MY_I_A(79)] = RCT[MY_I(79)] * V[MY_I(43)] * V[MY_I(57)];
	A[MY_I_A(80)] = RCT[MY_I(80)] * V[MY_I(39)] * V[MY_I(57)];
	A[MY_I_A(81)] = RCT[MY_I(81)] * V[MY_I(36)] * V[MY_I(57)];
	A[MY_I_A(82)] = RCT[MY_I(82)] * V[MY_I(48)] * V[MY_I(57)];
	A[MY_I_A(83)] = RCT[MY_I(83)] * V[MY_I(54)] * V[MY_I(57)];
	A[MY_I_A(84)] = RCT[MY_I(84)] * V[MY_I(44)] * V[MY_I(57)];
	A[MY_I_A(85)] = RCT[MY_I(85)] * V[MY_I(34)] * V[MY_I(57)];
	A[MY_I_A(86)] = RCT[MY_I(86)] * V[MY_I(35)] * V[MY_I(57)];
	A[MY_I_A(87)] = RCT[MY_I(87)] * V[MY_I(56)] * V[MY_I(57)];
	A[MY_I_A(88)] = RCT[MY_I(88)] * V[MY_I(49)] * V[MY_I(57)];
	A[MY_I_A(89)] = RCT[MY_I(89)] * V[MY_I(37)] * V[MY_I(57)];
	A[MY_I_A(90)] = RCT[MY_I(90)] * V[MY_I(42)] * V[MY_I(53)];
	A[MY_I_A(91)] = RCT[MY_I(91)] * V[MY_I(47)] * V[MY_I(53)];
	A[MY_I_A(92)] = RCT[MY_I(92)] * V[MY_I(30)] * V[MY_I(53)];
	A[MY_I_A(93)] = RCT[MY_I(93)] * V[MY_I(33)] * V[MY_I(53)];
	A[MY_I_A(94)] = RCT[MY_I(94)] * V[MY_I(29)] * V[MY_I(53)];
	A[MY_I_A(95)] = RCT[MY_I(95)] * V[MY_I(21)] * V[MY_I(53)];
	A[MY_I_A(96)] = RCT[MY_I(96)] * V[MY_I(23)] * V[MY_I(53)];
	A[MY_I_A(97)] = RCT[MY_I(97)] * V[MY_I(27)] * V[MY_I(53)];
	A[MY_I_A(98)] = RCT[MY_I(98)] * V[MY_I(28)] * V[MY_I(53)];
	A[MY_I_A(99)] = RCT[MY_I(99)] * V[MY_I(26)] * V[MY_I(53)];
	A[MY_I_A(100)] = RCT[MY_I(100)] * V[MY_I(23)] * V[MY_I(45)];
	A[MY_I_A(101)] = RCT[MY_I(101)] * V[MY_I(27)] * V[MY_I(45)];
	A[MY_I_A(102)] = RCT[MY_I(102)] * V[MY_I(28)] * V[MY_I(45)];
	A[MY_I_A(103)] = RCT[MY_I(103)] * V[MY_I(26)] * V[MY_I(45)];
	A[MY_I_A(104)] = RCT[MY_I(104)] * V[MY_I(50)] * V[MY_I(51)];
	A[MY_I_A(105)] = RCT[MY_I(105)] * V[MY_I(50)] * V[MY_I(56)];
	A[MY_I_A(106)] = RCT[MY_I(106)] * V[MY_I(50)] * V[MY_I(55)];
	A[MY_I_A(107)] = RCT[MY_I(107)] * V[MY_I(40)] * V[MY_I(50)];
	A[MY_I_A(108)] = RCT[MY_I(108)] * V[MY_I(43)] * V[MY_I(50)];
	A[MY_I_A(109)] = RCT[MY_I(109)] * V[MY_I(39)] * V[MY_I(50)];
	A[MY_I_A(110)] = RCT[MY_I(110)] * V[MY_I(36)] * V[MY_I(50)];
	A[MY_I_A(111)] = RCT[MY_I(111)] * V[MY_I(48)] * V[MY_I(50)];
	A[MY_I_A(112)] = RCT[MY_I(112)] * V[MY_I(49)] * V[MY_I(50)];
	A[MY_I_A(113)] = RCT[MY_I(113)] * V[MY_I(50)] * V[MY_I(54)];
	A[MY_I_A(114)] = RCT[MY_I(114)] * V[MY_I(34)] * V[MY_I(50)];
	A[MY_I_A(115)] = RCT[MY_I(115)] * V[MY_I(35)] * V[MY_I(50)];
	A[MY_I_A(116)] = RCT[MY_I(116)] * V[MY_I(44)] * V[MY_I(50)];
	A[MY_I_A(117)] = RCT[MY_I(117)] * V[MY_I(37)] * V[MY_I(50)];
	A[MY_I_A(118)] = RCT[MY_I(118)] * V[MY_I(51)] * V[MY_I(51)];
	A[MY_I_A(119)] = RCT[MY_I(119)] * V[MY_I(51)] * V[MY_I(56)];
	A[MY_I_A(120)] = RCT[MY_I(120)] * V[MY_I(51)] * V[MY_I(55)];
	A[MY_I_A(121)] = RCT[MY_I(121)] * V[MY_I(40)] * V[MY_I(51)];
	A[MY_I_A(122)] = RCT[MY_I(122)] * V[MY_I(43)] * V[MY_I(51)];
	A[MY_I_A(123)] = RCT[MY_I(123)] * V[MY_I(39)] * V[MY_I(51)];
	A[MY_I_A(124)] = RCT[MY_I(124)] * V[MY_I(36)] * V[MY_I(51)];
	A[MY_I_A(125)] = RCT[MY_I(125)] * V[MY_I(48)] * V[MY_I(51)];
	A[MY_I_A(126)] = RCT[MY_I(126)] * V[MY_I(49)] * V[MY_I(51)];
	A[MY_I_A(127)] = RCT[MY_I(127)] * V[MY_I(51)] * V[MY_I(54)];
	A[MY_I_A(128)] = RCT[MY_I(128)] * V[MY_I(34)] * V[MY_I(51)];
	A[MY_I_A(129)] = RCT[MY_I(129)] * V[MY_I(35)] * V[MY_I(51)];
	A[MY_I_A(130)] = RCT[MY_I(130)] * V[MY_I(44)] * V[MY_I(51)];
	A[MY_I_A(131)] = RCT[MY_I(131)] * V[MY_I(54)] * V[MY_I(56)];
	A[MY_I_A(132)] = RCT[MY_I(132)] * V[MY_I(54)] * V[MY_I(55)];
	A[MY_I_A(133)] = RCT[MY_I(133)] * V[MY_I(40)] * V[MY_I(54)];
	A[MY_I_A(134)] = RCT[MY_I(134)] * V[MY_I(43)] * V[MY_I(54)];
	A[MY_I_A(135)] = RCT[MY_I(135)] * V[MY_I(39)] * V[MY_I(54)];
	A[MY_I_A(136)] = RCT[MY_I(136)] * V[MY_I(36)] * V[MY_I(54)];
	A[MY_I_A(137)] = RCT[MY_I(137)] * V[MY_I(48)] * V[MY_I(54)];
	A[MY_I_A(138)] = RCT[MY_I(138)] * V[MY_I(49)] * V[MY_I(54)];
	A[MY_I_A(139)] = RCT[MY_I(139)] * V[MY_I(54)] * V[MY_I(54)];
	A[MY_I_A(140)] = RCT[MY_I(140)] * V[MY_I(34)] * V[MY_I(54)];
	A[MY_I_A(141)] = RCT[MY_I(141)] * V[MY_I(35)] * V[MY_I(54)];
	A[MY_I_A(142)] = RCT[MY_I(142)] * V[MY_I(44)] * V[MY_I(54)];
	A[MY_I_A(143)] = RCT[MY_I(143)] * V[MY_I(38)] * V[MY_I(50)];
	A[MY_I_A(144)] = RCT[MY_I(144)] * V[MY_I(38)] * V[MY_I(51)];
	A[MY_I_A(145)] = RCT[MY_I(145)] * V[MY_I(38)] * V[MY_I(54)];
	A[MY_I_A(146)] = RCT[MY_I(146)] * V[MY_I(38)] * V[MY_I(38)];
	A[MY_I_A(147)] = RCT[MY_I(147)] * V[MY_I(38)] * V[MY_I(57)];
	A[MY_I_A(148)] = RCT[MY_I(148)] * V[MY_I(31)] * V[MY_I(58)];
	A[MY_I_A(149)] = RCT[MY_I(149)] * V[MY_I(31)] * V[MY_I(50)];
	A[MY_I_A(150)] = RCT[MY_I(150)] * V[MY_I(31)] * V[MY_I(51)];
	A[MY_I_A(151)] = RCT[MY_I(151)] * V[MY_I(31)] * V[MY_I(54)];
	A[MY_I_A(152)] = RCT[MY_I(152)] * V[MY_I(31)] * V[MY_I(31)];
	A[MY_I_A(153)] = RCT[MY_I(153)] * V[MY_I(37)] * V[MY_I(51)];
	A[MY_I_A(154)] = RCT[MY_I(154)] * V[MY_I(37)] * V[MY_I(54)];
	A[MY_I_A(155)] = RCT[MY_I(155)] * V[MY_I(37)] * V[MY_I(37)];
	
	Vdot[MY_I(0)] = A[MY_I_A(47)];
	Vdot[MY_I(1)] = 0.4*A[MY_I_A(100)]+0.2*A[MY_I_A(101)]+0.06*A[MY_I_A(102)]+0.2*A[MY_I_A(103)];
	Vdot[MY_I(2)] = 0.2*A[MY_I_A(101)]+0.29*A[MY_I_A(102)]+0.2*A[MY_I_A(103)]+0.5*A[MY_I_A(127)]+0.5*A[MY_I_A(130)]+0.5
		   *A[MY_I_A(131)]+0.5*A[MY_I_A(132)]+0.5*A[MY_I_A(133)]+0.5*A[MY_I_A(134)]+0.5*A[MY_I_A(135)]+0.5
		   *A[MY_I_A(136)]+0.5*A[MY_I_A(137)]+0.5*A[MY_I_A(138)]+0.5*A[MY_I_A(154)];
	Vdot[MY_I(3)] = A[MY_I_A(48)];
	Vdot[MY_I(4)] = -A[MY_I_A(47)];
	Vdot[MY_I(5)] = -A[MY_I_A(50)];
	Vdot[MY_I(6)] = A[MY_I_A(1)]-A[MY_I_A(23)]-A[MY_I_A(24)];
	Vdot[MY_I(7)] = -A[MY_I_A(52)];
	Vdot[MY_I(8)] = -A[MY_I_A(53)];
	Vdot[MY_I(9)] = -A[MY_I_A(57)];
	Vdot[MY_I(10)] = -A[MY_I_A(58)];
	Vdot[MY_I(11)] = A[MY_I_A(74)]-A[MY_I_A(75)];
	Vdot[MY_I(12)] = -A[MY_I_A(3)]+A[MY_I_A(34)];
	Vdot[MY_I(13)] = -A[MY_I_A(8)]+A[MY_I_A(31)]+A[MY_I_A(32)];//-A[MY_I_A(33)];
	Vdot[MY_I(14)] = A[MY_I_A(40)]-A[MY_I_A(41)]-A[MY_I_A(42)];
	Vdot[MY_I(15)] = -A[MY_I_A(51)];
	Vdot[MY_I(16)] = -A[MY_I_A(49)]+0.06*A[MY_I_A(101)]+0.09*A[MY_I_A(102)];
	Vdot[MY_I(17)] = -A[MY_I_A(14)]-A[MY_I_A(68)]+A[MY_I_A(113)];
	Vdot[MY_I(18)] = A[MY_I_A(0)]+A[MY_I_A(2)]+A[MY_I_A(7)]-A[MY_I_A(21)]-A[MY_I_A(22)]+A[MY_I_A(23)];
	Vdot[MY_I(19)] = -A[MY_I_A(5)]+A[MY_I_A(29)]-A[MY_I_A(30)]-A[MY_I_A(45)];
	Vdot[MY_I(20)] = -A[MY_I_A(12)]-A[MY_I_A(66)]+A[MY_I_A(104)];
	Vdot[MY_I(21)] = 0.25*A[MY_I_A(57)]+0.17*A[MY_I_A(58)]-A[MY_I_A(59)]-0.5*A[MY_I_A(95)];
	Vdot[MY_I(22)] = -A[MY_I_A(69)]+A[MY_I_A(72)]-A[MY_I_A(73)];
	Vdot[MY_I(23)] = -A[MY_I_A(54)]-A[MY_I_A(96)]-A[MY_I_A(100)];
	Vdot[MY_I(24)] = -A[MY_I_A(4)]+A[MY_I_A(39)]+2*A[MY_I_A(42)]+A[MY_I_A(43)]-A[MY_I_A(44)]+A[MY_I_A(90)]+A[MY_I_A(91)]+A[MY_I_A(92)]+A[MY_I_A(93)]
			+A[MY_I_A(94)]+A[MY_I_A(95)];
	Vdot[MY_I(25)] = A[MY_I_A(9)]+A[MY_I_A(10)]+A[MY_I_A(11)]+1.87*A[MY_I_A(16)]+1.55*A[MY_I_A(17)]+A[MY_I_A(18)]-A[MY_I_A(48)]+A[MY_I_A(60)]+2
			*A[MY_I_A(63)]+A[MY_I_A(64)]+0.95*A[MY_I_A(84)]+A[MY_I_A(90)]+2*A[MY_I_A(92)]+A[MY_I_A(93)]+0.42*A[MY_I_A(100)]
			+0.33*A[MY_I_A(101)]+0.23*A[MY_I_A(102)]+0.33*A[MY_I_A(103)]+0.475*A[MY_I_A(130)]+0.95
			*A[MY_I_A(142)];
	Vdot[MY_I(26)] = -A[MY_I_A(71)]-A[MY_I_A(99)]-A[MY_I_A(103)];
	Vdot[MY_I(27)] = -A[MY_I_A(55)]-A[MY_I_A(97)]-A[MY_I_A(101)];
	Vdot[MY_I(28)] = -A[MY_I_A(56)]-A[MY_I_A(98)]-A[MY_I_A(102)];
	Vdot[MY_I(29)] = -A[MY_I_A(19)]-A[MY_I_A(65)]+0.7*A[MY_I_A(85)]+0.806*A[MY_I_A(86)]-A[MY_I_A(94)]+0.7*A[MY_I_A(128)]+0.806
			*A[MY_I_A(129)]+A[MY_I_A(140)]+A[MY_I_A(141)];
	Vdot[MY_I(30)] = -A[MY_I_A(16)]-A[MY_I_A(17)]-A[MY_I_A(63)]+0.89*A[MY_I_A(84)]+0.16*A[MY_I_A(85)]-A[MY_I_A(92)]+0.16*A[MY_I_A(128)]
			+0.445*A[MY_I_A(130)]+0.2*A[MY_I_A(140)]+0.89*A[MY_I_A(142)];
	Vdot[MY_I(31)] = A[MY_I_A(95)]-A[MY_I_A(148)]-A[MY_I_A(149)]-A[MY_I_A(150)]-A[MY_I_A(151)]-2*A[MY_I_A(152)];
	Vdot[MY_I(32)] = -A[MY_I_A(15)]+0.8*A[MY_I_A(20)]+0.025*A[MY_I_A(51)]-A[MY_I_A(62)]+0.25*A[MY_I_A(77)]+0.69*A[MY_I_A(78)]
			+1.06*A[MY_I_A(79)]+0.1*A[MY_I_A(82)]+0.1*A[MY_I_A(102)]+0.6*A[MY_I_A(120)]+0.75*A[MY_I_A(121)]
			+1.39*A[MY_I_A(122)]+0.55*A[MY_I_A(125)]+0.8*A[MY_I_A(132)]+0.86*A[MY_I_A(133)]+0.9*A[MY_I_A(134)]
			+0.55*A[MY_I_A(137)];
	Vdot[MY_I(33)] = -A[MY_I_A(18)]-A[MY_I_A(64)]+0.11*A[MY_I_A(84)]+0.17*A[MY_I_A(85)]+0.45*A[MY_I_A(86)]+A[MY_I_A(88)]-A[MY_I_A(93)]
			+0.75*A[MY_I_A(126)]+0.17*A[MY_I_A(128)]+0.45*A[MY_I_A(129)]+0.055*A[MY_I_A(130)]+A[MY_I_A(138)]
			+0.8*A[MY_I_A(140)]+A[MY_I_A(141)]+0.11*A[MY_I_A(142)];
	Vdot[MY_I(34)] = 0.75*A[MY_I_A(57)]-A[MY_I_A(85)]-A[MY_I_A(114)]-A[MY_I_A(128)]-A[MY_I_A(140)];
	Vdot[MY_I(35)] = 0.83*A[MY_I_A(58)]-A[MY_I_A(86)]-A[MY_I_A(115)]-A[MY_I_A(129)]-A[MY_I_A(141)];
	Vdot[MY_I(36)] = A[MY_I_A(55)]+A[MY_I_A(71)]-A[MY_I_A(81)]-A[MY_I_A(110)]-A[MY_I_A(124)]-A[MY_I_A(136)];
	Vdot[MY_I(37)] = -A[MY_I_A(89)]+A[MY_I_A(96)]+A[MY_I_A(97)]+A[MY_I_A(98)]+A[MY_I_A(99)]-A[MY_I_A(117)]-A[MY_I_A(153)]-A[MY_I_A(154)]-2
			*A[MY_I_A(155)];
	Vdot[MY_I(38)] = 0.25*A[MY_I_A(52)]+0.75*A[MY_I_A(53)]+0.9*A[MY_I_A(59)]+A[MY_I_A(69)]+2*A[MY_I_A(84)]+A[MY_I_A(130)]+2
			*A[MY_I_A(142)]-A[MY_I_A(143)]-A[MY_I_A(144)]-A[MY_I_A(145)]-2*A[MY_I_A(146)]-A[MY_I_A(147)];
	Vdot[MY_I(39)] = A[MY_I_A(54)]-A[MY_I_A(80)]-A[MY_I_A(109)]-A[MY_I_A(123)]-A[MY_I_A(135)];
	Vdot[MY_I(40)] = A[MY_I_A(52)]-A[MY_I_A(78)]-A[MY_I_A(107)]-A[MY_I_A(121)]-A[MY_I_A(133)];
	Vdot[MY_I(41)] = -A[MY_I_A(13)]-A[MY_I_A(67)]+A[MY_I_A(105)]+A[MY_I_A(106)]+A[MY_I_A(107)]+A[MY_I_A(108)]+A[MY_I_A(109)]+A[MY_I_A(110)]
			+A[MY_I_A(111)]+A[MY_I_A(112)]+A[MY_I_A(114)]+A[MY_I_A(115)]+A[MY_I_A(116)]+A[MY_I_A(143)]+A[MY_I_A(149)];
	Vdot[MY_I(42)] = -A[MY_I_A(9)]-A[MY_I_A(10)]+A[MY_I_A(12)]+0.13*A[MY_I_A(16)]+0.45*A[MY_I_A(17)]+0.009*A[MY_I_A(51)]-A[MY_I_A(60)]
			+0.5*A[MY_I_A(66)]+A[MY_I_A(69)]+A[MY_I_A(76)]+0.09*A[MY_I_A(77)]+0.04*A[MY_I_A(79)]+1.6*A[MY_I_A(80)]
			+A[MY_I_A(81)]+0.28*A[MY_I_A(82)]+A[MY_I_A(89)]-A[MY_I_A(90)]+A[MY_I_A(100)]+0.53*A[MY_I_A(101)]+0.18
			*A[MY_I_A(102)]+0.53*A[MY_I_A(103)]+1.5*A[MY_I_A(118)]+0.75*A[MY_I_A(119)]+0.75*A[MY_I_A(120)]+0.77
			*A[MY_I_A(121)]+0.8*A[MY_I_A(122)]+1.55*A[MY_I_A(123)]+1.25*A[MY_I_A(124)]+0.89*A[MY_I_A(125)]+0.75
			*A[MY_I_A(126)]+A[MY_I_A(127)]+A[MY_I_A(128)]+A[MY_I_A(129)]+0.5*A[MY_I_A(130)]+0.8*A[MY_I_A(135)]+0.5
			*A[MY_I_A(136)]+0.14*A[MY_I_A(137)]+A[MY_I_A(144)]+A[MY_I_A(150)]+1.75*A[MY_I_A(153)]+A[MY_I_A(154)]+2
			*A[MY_I_A(155)];

	Vdot[MY_I(43)] = A[MY_I_A(53)]-A[MY_I_A(79)]-A[MY_I_A(108)]-A[MY_I_A(122)]-A[MY_I_A(134)];
	Vdot[MY_I(44)] = A[MY_I_A(19)]+0.9*A[MY_I_A(59)]+A[MY_I_A(65)]-A[MY_I_A(74)]+A[MY_I_A(75)]-A[MY_I_A(84)]+A[MY_I_A(94)]-A[MY_I_A(116)]
			-A[MY_I_A(130)]-A[MY_I_A(142)];
	Vdot[MY_I(45)] = -A[MY_I_A(1)]-A[MY_I_A(2)]+A[MY_I_A(21)]-A[MY_I_A(25)]-A[MY_I_A(26)]-A[MY_I_A(27)]-A[MY_I_A(36)]-A[MY_I_A(100)]-A[MY_I_A(101)]
			-A[MY_I_A(102)]-A[MY_I_A(103)];
	Vdot[MY_I(46)] = -A[MY_I_A(20)]-A[MY_I_A(70)]+0.036*A[MY_I_A(77)]+0.08*A[MY_I_A(78)]+0.24*A[MY_I_A(79)]+A[MY_I_A(117)]
			+A[MY_I_A(148)];
	Vdot[MY_I(47)] = -A[MY_I_A(11)]+A[MY_I_A(13)]+0.2*A[MY_I_A(20)]+0.075*A[MY_I_A(51)]-A[MY_I_A(61)]+0.5*A[MY_I_A(67)]+0.75
			*A[MY_I_A(77)]+0.38*A[MY_I_A(78)]+0.35*A[MY_I_A(79)]+0.2*A[MY_I_A(80)]+A[MY_I_A(81)]+1.45*A[MY_I_A(82)]
			+A[MY_I_A(87)]+A[MY_I_A(89)]-A[MY_I_A(91)]+0.5*A[MY_I_A(101)]+0.72*A[MY_I_A(102)]+0.5*A[MY_I_A(103)]+0.75
			*A[MY_I_A(119)]+0.15*A[MY_I_A(120)]+0.41*A[MY_I_A(121)]+0.46*A[MY_I_A(122)]+0.35*A[MY_I_A(123)]
			+0.75*A[MY_I_A(124)]+0.725*A[MY_I_A(125)]+A[MY_I_A(131)]+0.2*A[MY_I_A(132)]+0.14*A[MY_I_A(133)]+0.1
			*A[MY_I_A(134)]+0.6*A[MY_I_A(135)]+A[MY_I_A(136)]+0.725*A[MY_I_A(137)]+A[MY_I_A(153)]+A[MY_I_A(154)]+2
			*A[MY_I_A(155)];
	Vdot[MY_I(48)] = A[MY_I_A(56)]-A[MY_I_A(82)]-A[MY_I_A(111)]-A[MY_I_A(125)]-A[MY_I_A(137)];
	Vdot[MY_I(49)] = A[MY_I_A(62)]-A[MY_I_A(88)]-A[MY_I_A(112)]-A[MY_I_A(126)]-A[MY_I_A(138)];
	Vdot[MY_I(50)] = 0.65*A[MY_I_A(5)]+2*A[MY_I_A(10)]+A[MY_I_A(11)]+A[MY_I_A(12)]+A[MY_I_A(13)]+0.8*A[MY_I_A(17)]+A[MY_I_A(18)]+A[MY_I_A(19)]
			+A[MY_I_A(20)]+A[MY_I_A(26)]-A[MY_I_A(27)]-A[MY_I_A(28)]-A[MY_I_A(29)]+A[MY_I_A(30)]-2*A[MY_I_A(31)]-2*A[MY_I_A(32)]+A[MY_I_A(33)]
			-A[MY_I_A(39)]-A[MY_I_A(46)]+A[MY_I_A(47)]+A[MY_I_A(48)]+0.17*A[MY_I_A(51)]+0.25*A[MY_I_A(57)]+0.17*A[MY_I_A(58)]
			+0.1*A[MY_I_A(59)]+A[MY_I_A(60)]+A[MY_I_A(63)]+A[MY_I_A(76)]+0.964*A[MY_I_A(77)]+0.92*A[MY_I_A(78)]+0.76
			*A[MY_I_A(79)]+A[MY_I_A(80)]+A[MY_I_A(81)]+A[MY_I_A(82)]+0.92*A[MY_I_A(84)]+A[MY_I_A(85)]+A[MY_I_A(86)]+A[MY_I_A(87)]+A[MY_I_A(88)]
			+A[MY_I_A(90)]+A[MY_I_A(92)]+0.12*A[MY_I_A(100)]+0.23*A[MY_I_A(101)]+0.26*A[MY_I_A(102)]+0.23
			*A[MY_I_A(103)]-A[MY_I_A(104)]-A[MY_I_A(105)]-A[MY_I_A(106)]-A[MY_I_A(107)]-A[MY_I_A(108)]-A[MY_I_A(109)]-A[MY_I_A(110)]
			-A[MY_I_A(111)]-A[MY_I_A(112)]-A[MY_I_A(113)]-A[MY_I_A(114)]-A[MY_I_A(115)]-A[MY_I_A(116)]-A[MY_I_A(117)]+A[MY_I_A(118)]
			+A[MY_I_A(119)]+A[MY_I_A(120)]+A[MY_I_A(121)]+A[MY_I_A(122)]+A[MY_I_A(123)]+A[MY_I_A(124)]+A[MY_I_A(125)]+A[MY_I_A(126)]
			+0.5*A[MY_I_A(127)]+2*A[MY_I_A(128)]+2*A[MY_I_A(129)]+0.46*A[MY_I_A(130)]+0.5*A[MY_I_A(131)]+0.5
			*A[MY_I_A(132)]+0.5*A[MY_I_A(133)]+0.5*A[MY_I_A(134)]+0.5*A[MY_I_A(135)]+0.5*A[MY_I_A(136)]+0.5
			*A[MY_I_A(137)]+0.5*A[MY_I_A(138)]+A[MY_I_A(140)]+A[MY_I_A(141)]+0.92*A[MY_I_A(142)]-A[MY_I_A(143)]+A[MY_I_A(144)]
			-A[MY_I_A(149)]+A[MY_I_A(150)]+0.5*A[MY_I_A(153)];
	Vdot[MY_I(51)] = A[MY_I_A(11)]+A[MY_I_A(14)]+A[MY_I_A(49)]+0.5*A[MY_I_A(66)]-A[MY_I_A(76)]+A[MY_I_A(83)]+0.22*A[MY_I_A(101)]+0.31
			*A[MY_I_A(102)]+0.22*A[MY_I_A(103)]-A[MY_I_A(104)]-2*A[MY_I_A(118)]-A[MY_I_A(119)]-A[MY_I_A(120)]-A[MY_I_A(121)]
			-A[MY_I_A(122)]-A[MY_I_A(123)]-A[MY_I_A(124)]-A[MY_I_A(125)]-A[MY_I_A(126)]-0.5*A[MY_I_A(127)]-A[MY_I_A(128)]
			-A[MY_I_A(129)]-A[MY_I_A(130)]+0.5*A[MY_I_A(131)]+0.5*A[MY_I_A(132)]+0.5*A[MY_I_A(133)]+0.5*A[MY_I_A(134)]
			+0.5*A[MY_I_A(135)]+0.5*A[MY_I_A(136)]+0.5*A[MY_I_A(137)]+0.5*A[MY_I_A(138)]+2*A[MY_I_A(139)]
			+A[MY_I_A(140)]+A[MY_I_A(141)]+A[MY_I_A(142)]-A[MY_I_A(144)]+A[MY_I_A(145)]-A[MY_I_A(150)]+A[MY_I_A(151)]-A[MY_I_A(153)]
			+0.5*A[MY_I_A(154)];
	Vdot[MY_I(52)] = A[MY_I_A(3)]+A[MY_I_A(4)]+0.35*A[MY_I_A(5)]+2*A[MY_I_A(8)]+A[MY_I_A(12)]+A[MY_I_A(13)]+A[MY_I_A(14)]+2*A[MY_I_A(24)]-A[MY_I_A(26)]
			+A[MY_I_A(27)]+A[MY_I_A(28)]-A[MY_I_A(33)]-A[MY_I_A(34)]-A[MY_I_A(43)]-A[MY_I_A(44)]-A[MY_I_A(45)]-A[MY_I_A(46)]-A[MY_I_A(47)]
			-A[MY_I_A(48)]-A[MY_I_A(49)]-A[MY_I_A(50)]-A[MY_I_A(51)]-A[MY_I_A(52)]-A[MY_I_A(53)]-A[MY_I_A(54)]-A[MY_I_A(55)]-A[MY_I_A(56)]
			-A[MY_I_A(57)]-A[MY_I_A(58)]-1.9*A[MY_I_A(59)]-A[MY_I_A(60)]-A[MY_I_A(61)]-A[MY_I_A(62)]-A[MY_I_A(63)]-A[MY_I_A(64)]-A[MY_I_A(65)]
			-0.5*A[MY_I_A(66)]-0.5*A[MY_I_A(67)]-A[MY_I_A(68)]-A[MY_I_A(69)]-A[MY_I_A(70)]-A[MY_I_A(71)]+0.1*A[MY_I_A(101)]
			+0.14*A[MY_I_A(102)]+0.1*A[MY_I_A(103)];
	Vdot[MY_I(53)] = 0.35*A[MY_I_A(5)]-A[MY_I_A(6)]-A[MY_I_A(7)]+A[MY_I_A(36)]-A[MY_I_A(37)]-A[MY_I_A(38)]-A[MY_I_A(39)]-A[MY_I_A(40)]+A[MY_I_A(41)]
			+A[MY_I_A(44)]+A[MY_I_A(69)]-A[MY_I_A(90)]-A[MY_I_A(91)]-A[MY_I_A(92)]-A[MY_I_A(93)]-A[MY_I_A(94)]-A[MY_I_A(95)]-A[MY_I_A(96)]
			-A[MY_I_A(97)]-A[MY_I_A(98)]-A[MY_I_A(99)];
	Vdot[MY_I(54)] = A[MY_I_A(15)]+A[MY_I_A(18)]+A[MY_I_A(61)]+A[MY_I_A(64)]+A[MY_I_A(68)]-A[MY_I_A(72)]+A[MY_I_A(73)]-A[MY_I_A(83)]+0.05*A[MY_I_A(84)]
			+A[MY_I_A(91)]+A[MY_I_A(93)]-A[MY_I_A(113)]-A[MY_I_A(127)]+0.025*A[MY_I_A(130)]-A[MY_I_A(131)]-A[MY_I_A(132)]
			-A[MY_I_A(133)]-A[MY_I_A(134)]-A[MY_I_A(135)]-A[MY_I_A(136)]-A[MY_I_A(137)]-A[MY_I_A(138)]-2*A[MY_I_A(139)]-A[MY_I_A(140)]
			-A[MY_I_A(141)]-0.95*A[MY_I_A(142)]-A[MY_I_A(145)]-A[MY_I_A(151)]-A[MY_I_A(154)];
	Vdot[MY_I(55)] = 0.83*A[MY_I_A(51)]+0.5*A[MY_I_A(67)]+A[MY_I_A(70)]-A[MY_I_A(77)]-A[MY_I_A(106)]-A[MY_I_A(120)]-A[MY_I_A(132)];
	Vdot[MY_I(56)] = A[MY_I_A(15)]+A[MY_I_A(50)]-A[MY_I_A(87)]-A[MY_I_A(105)]-A[MY_I_A(119)]-A[MY_I_A(131)];
	Vdot[MY_I(57)] = A[MY_I_A(0)]+A[MY_I_A(3)]+A[MY_I_A(6)]+A[MY_I_A(22)]-A[MY_I_A(25)]-A[MY_I_A(28)]-A[MY_I_A(34)]-2*A[MY_I_A(35)]-A[MY_I_A(37)]+A[MY_I_A(38)]
			-A[MY_I_A(76)]-A[MY_I_A(77)]-A[MY_I_A(78)]-A[MY_I_A(79)]-A[MY_I_A(80)]-A[MY_I_A(81)]-A[MY_I_A(82)]-A[MY_I_A(83)]-A[MY_I_A(84)]
			-A[MY_I_A(85)]-A[MY_I_A(86)]-A[MY_I_A(87)]-A[MY_I_A(88)]-A[MY_I_A(89)]-A[MY_I_A(147)];
	Vdot[MY_I(58)] = -A[MY_I_A(0)]+A[MY_I_A(4)]+0.65*A[MY_I_A(5)]+A[MY_I_A(7)]+A[MY_I_A(20)]-A[MY_I_A(22)]+A[MY_I_A(25)]+A[MY_I_A(28)]-A[MY_I_A(29)]
			+A[MY_I_A(30)]+2*A[MY_I_A(35)]-A[MY_I_A(36)]+2*A[MY_I_A(37)]-A[MY_I_A(40)]+A[MY_I_A(41)]-A[MY_I_A(43)]+A[MY_I_A(45)]+A[MY_I_A(70)]
			-A[MY_I_A(72)]+A[MY_I_A(73)]-A[MY_I_A(74)]+A[MY_I_A(75)]+A[MY_I_A(76)]+0.964*A[MY_I_A(77)]+0.92*A[MY_I_A(78)]+0.76
			*A[MY_I_A(79)]+A[MY_I_A(80)]+A[MY_I_A(81)]+A[MY_I_A(82)]+A[MY_I_A(83)]+A[MY_I_A(84)]+A[MY_I_A(85)]+A[MY_I_A(86)]+A[MY_I_A(87)]
			+A[MY_I_A(88)]+2*A[MY_I_A(89)]+A[MY_I_A(147)]-A[MY_I_A(148)]+A[MY_I_A(153)]+A[MY_I_A(154)]+2*A[MY_I_A(155)];
}

// Compute the function at current time
// using T as length
// Multi-GPU
void Fun() {
	Nfun++;
	int length;
	dim3 gridDim, blockDim;

	//timer_start(&metrics.odefun);
	cudaStartTimer(&timingfun, 0, 0);

	length = (39)*(19)*(39);

	// One-thread-per-cell decomposition
	// Register memory is the limiting factor
	blockDim.x = 128;
	gridDim.x = length / blockDim.x + (length % blockDim.x > 0);
	
	for(device = startDevice; device < endDevice; device++) {
		cudaSetDevice(device);

		dev_Fun<<<gridDim, blockDim>>>(
			length, 
			DataDevice[device].d_VAR, DataDevice[device].d_FIX, 
			DataDevice[device].d_RCONST, DataDevice[device].Fcn0);
	}
	
	
	for(device = startDevice; device < endDevice; device++) {
		cudaSetDevice(device);
	
		cudaDeviceSynchronize();
	}

	//timer_stop(&metrics.odefun);
	cudaStopTimer(&timingfun, 0, 0);
}

// Compute the function at current time
// using Tau as length
// Single-GPU
void FunTau() {
	Nfun++;
	int length;
	dim3 gridDim, blockDim;

	//timer_start(&metrics.odefun);

	length = (39)*(19)*(39);

	// One-thread-per-cell decomposition
	// Register memory is the limiting factor
	blockDim.x = 128;
	gridDim.x = length / blockDim.x + (length % blockDim.x > 0);
	
	cudaSetDevice(0);

	dev_Fun<<<gridDim, blockDim>>>(
		length, 
		Ynew, DataDevice[0].d_FIX, 
		DataDevice[0].d_RCONST, Fcn);
	
	
	for(device = startDevice; device < endDevice; device++) {
		cudaSetDevice(device);
	
		cudaDeviceSynchronize();
	}

	//timer_stop(&metrics.odefun);
}

__global__ void dev_WSCAL(uint32_t N, double Alpha, double * X) {
	int idx = blockDim.x*blockIdx.x + threadIdx.x;

	if(idx < N) {
		X[idx] *= Alpha;
	}
}

__global__ void dev_WNEG(uint32_t N, double * X) {
        int idx = blockDim.x*blockIdx.x + threadIdx.x;

        if(idx < N) {
                X[idx] = - X[idx] ;
        }
}

__host__ void WSCAL(uint32_t N, double Alpha, double * d_X) {
	dim3 gridDim, blockDim;

	if(Alpha == 1) {
		return;
	} else if(Alpha == 0) {
		cudaMemset(d_X, 0, N*sizeof(double));
	} else {
		// 512 threads per block
		blockDim.x = 512;

		gridDim.x = N / blockDim.x + (N % blockDim.x > 0);
                if ( Alpha == - ONE ) {
			dev_WNEG<<<gridDim, blockDim>>>(N, d_X);
                } else {
			dev_WSCAL<<<gridDim, blockDim>>>(N, Alpha, d_X);
                }
	}
}

__global__ void dev_WAXPY(uint32_t N, double Alpha, double * X, double * Y) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx < N) {
		Y[idx] += Alpha * X[idx];
	}
}

__host__ void WAXPY(uint32_t N, double Alpha, double * d_X, double * d_Y) {
	dim3 gridDim, blockDim;

	if(Alpha == 0) {
		return;
	} else if(d_X == d_Y) {
		WSCAL(N, 1+Alpha, d_X);
	} else {
		// 512 threads per block
		blockDim.x = 512;
		gridDim.x = N / blockDim.x + (N % blockDim.x > 0);
		dev_WAXPY<<<gridDim, blockDim>>>(N, Alpha, d_X, d_Y);
	}
}

__global__ void dev_WCOPY(uint32_t N, double * X, double * Y)
{
        uint32_t idx = blockDim.x*blockIdx.x + threadIdx.x;

        if(idx < N) {
                Y[idx] = X[idx];
        }
}

__host__ void WCOPY(uint32_t N, double * d_X, double * d_Y) {
	dim3 gridDim, blockDim;

	if(d_X == d_Y) return;

	// 512 threads per block
	blockDim.x = 512;
	gridDim.x = N / blockDim.x + (N % blockDim.x > 0);

	cudaMemcpy(d_Y, d_X, N*sizeof(double), cudaMemcpyDeviceToDevice);
}

__global__ void dev_WAXPY_HC(
	int dom_dim, int * mask, int * mask2,
	double ros_C, int Direction, double * vecH, double * K1, double * Ko) {

	int i ;
	int p = blockIdx.x*blockDim.x + threadIdx.x;
	if(p >= dom_dim) return;
	if ( mask[p] == 0 || mask2[p] == 0 ) return ;

	for ( i = 0 ; i < NVAR ; i++ ) {
		Ko[MY_I(i)] += K1[MY_I(i)] * ( ros_C / (Direction*vecH[p]) ) ;
	}
}

__host__ void WAXPY_HC(
	int length, int *mask, int *mask2, 
	double ros_C, int Direction, double * vecH, double * K1, double * Ko) {

        dim3 gridDim, blockDim;

        blockDim.x = 128;
        gridDim.x = length / blockDim.x + (length % blockDim.x > 0);

        dev_WAXPY_HC<<<gridDim, blockDim>>>(length, mask, mask2, ros_C, Direction, vecH, K1, Ko ) ;
} // end function  WAXPY

void ros_FunTimeDerivative(
        double T, double Roundoff,
	double * Fcn0, double dFdT[])
{
	Nfun++;
	int length;
	dim3 gridDim, blockDim;
	
	length = (39)*(19)*(39);
	
	// One-thread-per-cell decomposition
	// Register memory is the limiting factor
	blockDim.x = 128;
	gridDim.x = length / blockDim.x + (length % blockDim.x > 0);

	double Delta;
	int length2 = NVAR * 39 * 39 * 19;

	Delta = SQRT(Roundoff) * MAX(DeltaMin,ABS(T));
	cudaSetDevice(0);
	dev_Fun<<<gridDim, blockDim>>>(
			length, 
			DataDevice[0].d_VAR, DataDevice[0].d_FIX, 
			DataDevice[0].d_RCONST, dFdT);

	WAXPY(length2, (-ONE), Fcn0, dFdT);
	WSCAL(length2, (ONE / Delta), dFdT);
}

__global__ void dev_moreToGo_timeloop(
	int length, 
	double T, double Tend, 
	double Roundoff, int Direction, 
	double *vecT, double *vecH, 
	int * vecMask) {

	int i = threadIdx.x;
	int j = blockIdx.y;
	int k = blockIdx.x;
	int p = j*gridDim.x*blockDim.x + k*blockDim.x + i;
	if ( p >= length ) return ;

	vecMask[p] = (( (Direction > 0) && ((vecT[p]-Tend)+Roundoff <= 0.0) ) || 
			( (Direction < 0) && ((Tend-vecT[p])+Roundoff <= 0.0) ));

	if(vecMask[p]) {
		vecH[p] = MIN(vecH[p],ABS(Tend-vecT[p]));
	}
}

__host__ int moreToGo_timeloop(
	double T, double Tend, double Roundoff, int Direction,
	double * d_vecT, double * d_vecH, int * d_vecMask) {

        int retval ;
	dim3 gridDim, blockDim;
        int i ;
        // One-thread-per-cell decomposition
        // Multiple threads per block
        gridDim.y  = 39;
        gridDim.x  = 19;
        blockDim.x = 39;

        int length  = gridDim.y*gridDim.x*blockDim.x;

 	dev_moreToGo_timeloop <<<gridDim, blockDim>>>(length, T, Tend, Roundoff, Direction, d_vecT, d_vecH, d_vecMask);

        // Get return values
	int * vecMask = (int*)malloc(length*sizeof(int));
        cudaMemcpy(vecMask, d_vecMask, length*sizeof(int), cudaMemcpyDeviceToHost);

        retval = 0 ;
        for(i = 0; i < length; i++) {
		if(vecMask[i] != 0) { retval = 1 ; break ; }
        }

        int cnt = 0 ;
        for(i = 0; i < length; i++) {
		if(vecMask[i] != 0) {cnt++;}
        }
        free(vecMask) ;

        double * tmp = (double *)malloc(length*sizeof(double) ) ;
        cudaMemcpy(tmp, d_vecT, length*sizeof(int), cudaMemcpyDeviceToHost);
        double minT =  999999.;
        double maxT = -999999.;
        for(i = 0; i < length; i++) {
		minT = MIN(minT,tmp[i]);
		maxT = MAX(maxT,tmp[i]);
        }
        cudaMemcpy(tmp, d_vecH, length*sizeof(int), cudaMemcpyDeviceToHost);
        double minH =  999999.;
        double maxH = -999999.;
        for ( i = 0 ; i < length ; i++ ) {
		minH = MIN(minH,tmp[i]) ;
		maxH = MAX(maxH,tmp[i]) ;
        }

#ifndef SUPPRESS_STATS_OUTPUT
        fprintf(stderr,"moreToGo returns %d.  Still active %d. minT %E maxT %E minH %E maxH %E\n",retval,cnt,minT,maxT,minH,maxH) ;
#endif

        return( retval ) ;

}

__global__ void dev_Jac_SP(int length, double * V, double * F, double * RCT, double * JVS) {

	int p = blockIdx.x*blockDim.x + threadIdx.x;

	if(p >= length) return;

        //if ( vecMask[p] == 0 ) return;

	double B[280];

	#define MY_I_A(i) MY_I(i)
	#define MY_I(i) (i)

	B[MY_I_A(0)] = RCT[MY_I(0)];
	B[MY_I_A(1)] = RCT[MY_I(1)];
	B[MY_I_A(2)] = RCT[MY_I(2)];
	B[MY_I_A(3)] = RCT[MY_I(3)];
	B[MY_I_A(4)] = RCT[MY_I(4)];
	B[MY_I_A(5)] = RCT[MY_I(5)];
	B[MY_I_A(6)] = RCT[MY_I(6)];
	B[MY_I_A(7)] = RCT[MY_I(7)];
	B[MY_I_A(8)] = RCT[MY_I(8)];
	B[MY_I_A(9)] = RCT[MY_I(9)];
	B[MY_I_A(10)] = RCT[MY_I(10)];
	B[MY_I_A(11)] = RCT[MY_I(11)];
	B[MY_I_A(12)] = RCT[MY_I(12)];
	B[MY_I_A(13)] = RCT[MY_I(13)];
	B[MY_I_A(14)] = RCT[MY_I(14)];
	B[MY_I_A(15)] = RCT[MY_I(15)];
	B[MY_I_A(16)] = RCT[MY_I(16)];
	B[MY_I_A(17)] = RCT[MY_I(17)];
	B[MY_I_A(18)] = RCT[MY_I(18)];
	B[MY_I_A(19)] = RCT[MY_I(19)];
	B[MY_I_A(20)] = RCT[MY_I(20)];
	B[MY_I_A(21)] = RCT[MY_I(21)] * F[MY_I(1)];
	B[MY_I_A(23)] = RCT[MY_I(22)] * V[MY_I(58)];
	B[MY_I_A(24)] = RCT[MY_I(22)] * V[MY_I(18)];
	B[MY_I_A(25)] = RCT[MY_I(23)] * F[MY_I(1)];
	B[MY_I_A(27)] = RCT[MY_I(24)] * F[MY_I(0)];
	B[MY_I_A(29)] = RCT[MY_I(25)] * V[MY_I(57)];
	B[MY_I_A(30)] = RCT[MY_I(25)] * V[MY_I(45)];
	B[MY_I_A(31)] = RCT[MY_I(26)] * V[MY_I(52)];
	B[MY_I_A(32)] = RCT[MY_I(26)] * V[MY_I(45)];
	B[MY_I_A(33)] = RCT[MY_I(27)] * V[MY_I(50)];
	B[MY_I_A(34)] = RCT[MY_I(27)] * V[MY_I(45)];
	B[MY_I_A(35)] = RCT[MY_I(28)] * V[MY_I(57)];
	B[MY_I_A(36)] = RCT[MY_I(28)] * V[MY_I(50)];
	B[MY_I_A(37)] = RCT[MY_I(29)] * V[MY_I(58)];
	B[MY_I_A(38)] = RCT[MY_I(29)] * V[MY_I(50)];
	B[MY_I_A(39)] = RCT[MY_I(30)];
	B[MY_I_A(40)] = RCT[MY_I(31)] * 2* V [MY_I(50)];
	B[MY_I_A(41)] = RCT[MY_I(32)] * 2* V [MY_I(50)] * F[MY_I(0)];
	B[MY_I_A(43)] = RCT[MY_I(33)] * V[MY_I(52)];
	B[MY_I_A(44)] = RCT[MY_I(33)] * V[MY_I(13)];
	B[MY_I_A(45)] = RCT[MY_I(34)] * V[MY_I(57)];
	B[MY_I_A(46)] = RCT[MY_I(34)] * V[MY_I(52)];
	B[MY_I_A(47)] = RCT[MY_I(35)] * 2* V [MY_I(57)] * F[MY_I(1)];
	B[MY_I_A(49)] = RCT[MY_I(36)] * V[MY_I(58)];
	B[MY_I_A(50)] = RCT[MY_I(36)] * V[MY_I(45)];
	B[MY_I_A(51)] = RCT[MY_I(37)] * V[MY_I(57)];
	B[MY_I_A(52)] = RCT[MY_I(37)] * V[MY_I(53)];
	B[MY_I_A(53)] = RCT[MY_I(38)] * V[MY_I(58)];
	B[MY_I_A(54)] = RCT[MY_I(38)] * V[MY_I(53)];
	B[MY_I_A(55)] = RCT[MY_I(39)] * V[MY_I(53)];
	B[MY_I_A(56)] = RCT[MY_I(39)] * V[MY_I(50)];
	B[MY_I_A(57)] = RCT[MY_I(40)] * V[MY_I(58)];
	B[MY_I_A(58)] = RCT[MY_I(40)] * V[MY_I(53)];
	B[MY_I_A(59)] = RCT[MY_I(41)];
	B[MY_I_A(60)] = RCT[MY_I(42)];
	B[MY_I_A(61)] = RCT[MY_I(43)] * V[MY_I(58)];
	B[MY_I_A(62)] = RCT[MY_I(43)] * V[MY_I(52)];
	B[MY_I_A(63)] = RCT[MY_I(44)] * V[MY_I(52)];
	B[MY_I_A(64)] = RCT[MY_I(44)] * V[MY_I(24)];
	B[MY_I_A(65)] = RCT[MY_I(45)] * V[MY_I(52)];
	B[MY_I_A(66)] = RCT[MY_I(45)] * V[MY_I(19)];
	B[MY_I_A(67)] = RCT[MY_I(46)] * V[MY_I(52)];
	B[MY_I_A(68)] = RCT[MY_I(46)] * V[MY_I(50)];
	B[MY_I_A(69)] = RCT[MY_I(47)] * V[MY_I(52)];
	B[MY_I_A(70)] = RCT[MY_I(47)] * V[MY_I(4)];
	B[MY_I_A(71)] = RCT[MY_I(48)] * V[MY_I(52)];
	B[MY_I_A(72)] = RCT[MY_I(48)] * V[MY_I(25)];
	B[MY_I_A(73)] = RCT[MY_I(49)] * V[MY_I(52)];
	B[MY_I_A(74)] = RCT[MY_I(49)] * V[MY_I(16)];
	B[MY_I_A(75)] = RCT[MY_I(50)] * V[MY_I(52)];
	B[MY_I_A(76)] = RCT[MY_I(50)] * V[MY_I(5)];
	B[MY_I_A(77)] = RCT[MY_I(51)] * V[MY_I(52)];
	B[MY_I_A(78)] = RCT[MY_I(51)] * V[MY_I(15)];
	B[MY_I_A(79)] = RCT[MY_I(52)] * V[MY_I(52)];
	B[MY_I_A(80)] = RCT[MY_I(52)] * V[MY_I(7)];
	B[MY_I_A(81)] = RCT[MY_I(53)] * V[MY_I(52)];
	B[MY_I_A(82)] = RCT[MY_I(53)] * V[MY_I(8)];
	B[MY_I_A(83)] = RCT[MY_I(54)] * V[MY_I(52)];
	B[MY_I_A(84)] = RCT[MY_I(54)] * V[MY_I(23)];
	B[MY_I_A(85)] = RCT[MY_I(55)] * V[MY_I(52)];
	B[MY_I_A(86)] = RCT[MY_I(55)] * V[MY_I(27)];
	B[MY_I_A(87)] = RCT[MY_I(56)] * V[MY_I(52)];
	B[MY_I_A(88)] = RCT[MY_I(56)] * V[MY_I(28)];
	B[MY_I_A(89)] = RCT[MY_I(57)] * V[MY_I(52)];
	B[MY_I_A(90)] = RCT[MY_I(57)] * V[MY_I(9)];
	B[MY_I_A(91)] = RCT[MY_I(58)] * V[MY_I(52)];
	B[MY_I_A(92)] = RCT[MY_I(58)] * V[MY_I(10)];
	B[MY_I_A(93)] = RCT[MY_I(59)] * V[MY_I(52)];
	B[MY_I_A(94)] = RCT[MY_I(59)] * V[MY_I(21)];
	B[MY_I_A(95)] = RCT[MY_I(60)] * V[MY_I(52)];
	B[MY_I_A(96)] = RCT[MY_I(60)] * V[MY_I(42)];
	B[MY_I_A(97)] = RCT[MY_I(61)] * V[MY_I(52)];
	B[MY_I_A(98)] = RCT[MY_I(61)] * V[MY_I(47)];
	B[MY_I_A(99)] = RCT[MY_I(62)] * V[MY_I(52)];
	B[MY_I_A(100)] = RCT[MY_I(62)] * V[MY_I(32)];
	B[MY_I_A(101)] = RCT[MY_I(63)] * V[MY_I(52)];
	B[MY_I_A(102)] = RCT[MY_I(63)] * V[MY_I(30)];
	B[MY_I_A(103)] = RCT[MY_I(64)] * V[MY_I(52)];
	B[MY_I_A(104)] = RCT[MY_I(64)] * V[MY_I(33)];
	B[MY_I_A(105)] = RCT[MY_I(65)] * V[MY_I(52)];
	B[MY_I_A(106)] = RCT[MY_I(65)] * V[MY_I(29)];
	B[MY_I_A(107)] = RCT[MY_I(66)] * V[MY_I(52)];
	B[MY_I_A(108)] = RCT[MY_I(66)] * V[MY_I(20)];
	B[MY_I_A(109)] = RCT[MY_I(67)] * V[MY_I(52)];
	B[MY_I_A(110)] = RCT[MY_I(67)] * V[MY_I(41)];
	B[MY_I_A(111)] = RCT[MY_I(68)] * V[MY_I(52)];
	B[MY_I_A(112)] = RCT[MY_I(68)] * V[MY_I(17)];
	B[MY_I_A(113)] = RCT[MY_I(69)] * V[MY_I(52)];
	B[MY_I_A(114)] = RCT[MY_I(69)] * V[MY_I(22)];
	B[MY_I_A(115)] = RCT[MY_I(70)] * V[MY_I(52)];
	B[MY_I_A(116)] = RCT[MY_I(70)] * V[MY_I(46)];
	B[MY_I_A(117)] = RCT[MY_I(71)] * V[MY_I(52)];
	B[MY_I_A(118)] = RCT[MY_I(71)] * V[MY_I(26)];
	B[MY_I_A(119)] = RCT[MY_I(72)] * V[MY_I(58)];
	B[MY_I_A(120)] = RCT[MY_I(72)] * V[MY_I(54)];
	B[MY_I_A(121)] = RCT[MY_I(73)];
	B[MY_I_A(122)] = RCT[MY_I(74)] * V[MY_I(58)];
	B[MY_I_A(123)] = RCT[MY_I(74)] * V[MY_I(44)];
	B[MY_I_A(124)] = RCT[MY_I(75)];
	B[MY_I_A(125)] = RCT[MY_I(76)] * V[MY_I(57)];
	B[MY_I_A(126)] = RCT[MY_I(76)] * V[MY_I(51)];
	B[MY_I_A(127)] = RCT[MY_I(77)] * V[MY_I(57)];
	B[MY_I_A(128)] = RCT[MY_I(77)] * V[MY_I(55)];
	B[MY_I_A(129)] = RCT[MY_I(78)] * V[MY_I(57)];
	B[MY_I_A(130)] = RCT[MY_I(78)] * V[MY_I(40)];
	B[MY_I_A(131)] = RCT[MY_I(79)] * V[MY_I(57)];
	B[MY_I_A(132)] = RCT[MY_I(79)] * V[MY_I(43)];
	B[MY_I_A(133)] = RCT[MY_I(80)] * V[MY_I(57)];
	B[MY_I_A(134)] = RCT[MY_I(80)] * V[MY_I(39)];
	B[MY_I_A(135)] = RCT[MY_I(81)] * V[MY_I(57)];
	B[MY_I_A(136)] = RCT[MY_I(81)] * V[MY_I(36)];
	B[MY_I_A(137)] = RCT[MY_I(82)] * V[MY_I(57)];
	B[MY_I_A(138)] = RCT[MY_I(82)] * V[MY_I(48)];
	B[MY_I_A(139)] = RCT[MY_I(83)] * V[MY_I(57)];
	B[MY_I_A(140)] = RCT[MY_I(83)] * V[MY_I(54)];
	B[MY_I_A(141)] = RCT[MY_I(84)] * V[MY_I(57)];
	B[MY_I_A(142)] = RCT[MY_I(84)] * V[MY_I(44)];
	B[MY_I_A(143)] = RCT[MY_I(85)] * V[MY_I(57)];
	B[MY_I_A(144)] = RCT[MY_I(85)] * V[MY_I(34)];
	B[MY_I_A(145)] = RCT[MY_I(86)] * V[MY_I(57)];
	B[MY_I_A(146)] = RCT[MY_I(86)] * V[MY_I(35)];
	B[MY_I_A(147)] = RCT[MY_I(87)] * V[MY_I(57)];
	B[MY_I_A(148)] = RCT[MY_I(87)] * V[MY_I(56)];
	B[MY_I_A(149)] = RCT[MY_I(88)] * V[MY_I(57)];
	B[MY_I_A(150)] = RCT[MY_I(88)] * V[MY_I(49)];
	B[MY_I_A(151)] = RCT[MY_I(89)] * V[MY_I(57)];
	B[MY_I_A(152)] = RCT[MY_I(89)] * V[MY_I(37)];
	B[MY_I_A(153)] = RCT[MY_I(90)] * V[MY_I(53)];
	B[MY_I_A(154)] = RCT[MY_I(90)] * V[MY_I(42)];
	B[MY_I_A(155)] = RCT[MY_I(91)] * V[MY_I(53)];
	B[MY_I_A(156)] = RCT[MY_I(91)] * V[MY_I(47)];
	B[MY_I_A(157)] = RCT[MY_I(92)] * V[MY_I(53)];
	B[MY_I_A(158)] = RCT[MY_I(92)] * V[MY_I(30)];
	B[MY_I_A(159)] = RCT[MY_I(93)] * V[MY_I(53)];
	B[MY_I_A(160)] = RCT[MY_I(93)] * V[MY_I(33)];
	B[MY_I_A(161)] = RCT[MY_I(94)] * V[MY_I(53)];
	B[MY_I_A(162)] = RCT[MY_I(94)] * V[MY_I(29)];
	B[MY_I_A(163)] = RCT[MY_I(95)] * V[MY_I(53)];
	B[MY_I_A(164)] = RCT[MY_I(95)] * V[MY_I(21)];
	B[MY_I_A(165)] = RCT[MY_I(96)] * V[MY_I(53)];
	B[MY_I_A(166)] = RCT[MY_I(96)] * V[MY_I(23)];
	B[MY_I_A(167)] = RCT[MY_I(97)] * V[MY_I(53)];
	B[MY_I_A(168)] = RCT[MY_I(97)] * V[MY_I(27)];
	B[MY_I_A(169)] = RCT[MY_I(98)] * V[MY_I(53)];
	B[MY_I_A(170)] = RCT[MY_I(98)] * V[MY_I(28)];
	B[MY_I_A(171)] = RCT[MY_I(99)] * V[MY_I(53)];
	B[MY_I_A(172)] = RCT[MY_I(99)] * V[MY_I(26)];
	B[MY_I_A(173)] = RCT[MY_I(100)] * V[MY_I(45)];
	B[MY_I_A(174)] = RCT[MY_I(100)] * V[MY_I(23)];
	B[MY_I_A(175)] = RCT[MY_I(101)] * V[MY_I(45)];
	B[MY_I_A(176)] = RCT[MY_I(101)] * V[MY_I(27)];
	B[MY_I_A(177)] = RCT[MY_I(102)] * V[MY_I(45)];
	B[MY_I_A(178)] = RCT[MY_I(102)] * V[MY_I(28)];
	B[MY_I_A(179)] = RCT[MY_I(103)] * V[MY_I(45)];
	B[MY_I_A(180)] = RCT[MY_I(103)] * V[MY_I(26)];
	B[MY_I_A(181)] = RCT[MY_I(104)] * V[MY_I(51)];
	B[MY_I_A(182)] = RCT[MY_I(104)] * V[MY_I(50)];
	B[MY_I_A(183)] = RCT[MY_I(105)] * V[MY_I(56)];
	B[MY_I_A(184)] = RCT[MY_I(105)] * V[MY_I(50)];
	B[MY_I_A(185)] = RCT[MY_I(106)] * V[MY_I(55)];
	B[MY_I_A(186)] = RCT[MY_I(106)] * V[MY_I(50)];
	B[MY_I_A(187)] = RCT[MY_I(107)] * V[MY_I(50)];
	B[MY_I_A(188)] = RCT[MY_I(107)] * V[MY_I(40)];
	B[MY_I_A(189)] = RCT[MY_I(108)] * V[MY_I(50)];
	B[MY_I_A(190)] = RCT[MY_I(108)] * V[MY_I(43)];
	B[MY_I_A(191)] = RCT[MY_I(109)] * V[MY_I(50)];
	B[MY_I_A(192)] = RCT[MY_I(109)] * V[MY_I(39)];
	B[MY_I_A(193)] = RCT[MY_I(110)] * V[MY_I(50)];
	B[MY_I_A(194)] = RCT[MY_I(110)] * V[MY_I(36)];
	B[MY_I_A(195)] = RCT[MY_I(111)] * V[MY_I(50)];
	B[MY_I_A(196)] = RCT[MY_I(111)] * V[MY_I(48)];
	B[MY_I_A(197)] = RCT[MY_I(112)] * V[MY_I(50)];
	B[MY_I_A(198)] = RCT[MY_I(112)] * V[MY_I(49)];
	B[MY_I_A(199)] = RCT[MY_I(113)] * V[MY_I(54)];
	B[MY_I_A(200)] = RCT[MY_I(113)] * V[MY_I(50)];
	B[MY_I_A(201)] = RCT[MY_I(114)] * V[MY_I(50)];
	B[MY_I_A(202)] = RCT[MY_I(114)] * V[MY_I(34)];
	B[MY_I_A(203)] = RCT[MY_I(115)] * V[MY_I(50)];
	B[MY_I_A(204)] = RCT[MY_I(115)] * V[MY_I(35)];
	B[MY_I_A(205)] = RCT[MY_I(116)] * V[MY_I(50)];
	B[MY_I_A(206)] = RCT[MY_I(116)] * V[MY_I(44)];
	B[MY_I_A(207)] = RCT[MY_I(117)] * V[MY_I(50)];
	B[MY_I_A(208)] = RCT[MY_I(117)] * V[MY_I(37)];
	B[MY_I_A(209)] = RCT[MY_I(118)] * 2* V [MY_I(51)];
	B[MY_I_A(210)] = RCT[MY_I(119)] * V[MY_I(56)];
	B[MY_I_A(211)] = RCT[MY_I(119)] * V[MY_I(51)];
	B[MY_I_A(212)] = RCT[MY_I(120)] * V[MY_I(55)];
	B[MY_I_A(213)] = RCT[MY_I(120)] * V[MY_I(51)];
	B[MY_I_A(214)] = RCT[MY_I(121)] * V[MY_I(51)];
	B[MY_I_A(215)] = RCT[MY_I(121)] * V[MY_I(40)];
	B[MY_I_A(216)] = RCT[MY_I(122)] * V[MY_I(51)];
	B[MY_I_A(217)] = RCT[MY_I(122)] * V[MY_I(43)];
	B[MY_I_A(218)] = RCT[MY_I(123)] * V[MY_I(51)];
	B[MY_I_A(219)] = RCT[MY_I(123)] * V[MY_I(39)];
	B[MY_I_A(220)] = RCT[MY_I(124)] * V[MY_I(51)];
	B[MY_I_A(221)] = RCT[MY_I(124)] * V[MY_I(36)];
	B[MY_I_A(222)] = RCT[MY_I(125)] * V[MY_I(51)];
	B[MY_I_A(223)] = RCT[MY_I(125)] * V[MY_I(48)];
	B[MY_I_A(224)] = RCT[MY_I(126)] * V[MY_I(51)];
	B[MY_I_A(225)] = RCT[MY_I(126)] * V[MY_I(49)];
	B[MY_I_A(226)] = RCT[MY_I(127)] * V[MY_I(54)];
	B[MY_I_A(227)] = RCT[MY_I(127)] * V[MY_I(51)];
	B[MY_I_A(228)] = RCT[MY_I(128)] * V[MY_I(51)];
	B[MY_I_A(229)] = RCT[MY_I(128)] * V[MY_I(34)];
	B[MY_I_A(230)] = RCT[MY_I(129)] * V[MY_I(51)];
	B[MY_I_A(231)] = RCT[MY_I(129)] * V[MY_I(35)];
	B[MY_I_A(232)] = RCT[MY_I(130)] * V[MY_I(51)];
	B[MY_I_A(233)] = RCT[MY_I(130)] * V[MY_I(44)];
	B[MY_I_A(234)] = RCT[MY_I(131)] * V[MY_I(56)];
	B[MY_I_A(235)] = RCT[MY_I(131)] * V[MY_I(54)];
	B[MY_I_A(236)] = RCT[MY_I(132)] * V[MY_I(55)];
	B[MY_I_A(237)] = RCT[MY_I(132)] * V[MY_I(54)];
	B[MY_I_A(238)] = RCT[MY_I(133)] * V[MY_I(54)];
	B[MY_I_A(239)] = RCT[MY_I(133)] * V[MY_I(40)];
	B[MY_I_A(240)] = RCT[MY_I(134)] * V[MY_I(54)];
	B[MY_I_A(241)] = RCT[MY_I(134)] * V[MY_I(43)];
	B[MY_I_A(242)] = RCT[MY_I(135)] * V[MY_I(54)];
	B[MY_I_A(243)] = RCT[MY_I(135)] * V[MY_I(39)];
	B[MY_I_A(244)] = RCT[MY_I(136)] * V[MY_I(54)];
	B[MY_I_A(245)] = RCT[MY_I(136)] * V[MY_I(36)];
	B[MY_I_A(246)] = RCT[MY_I(137)] * V[MY_I(54)];
	B[MY_I_A(247)] = RCT[MY_I(137)] * V[MY_I(48)];
	B[MY_I_A(248)] = RCT[MY_I(138)] * V[MY_I(54)];
	B[MY_I_A(249)] = RCT[MY_I(138)] * V[MY_I(49)];
	B[MY_I_A(250)] = RCT[MY_I(139)] * 2* V [MY_I(54)];
	B[MY_I_A(251)] = RCT[MY_I(140)] * V[MY_I(54)];
	B[MY_I_A(252)] = RCT[MY_I(140)] * V[MY_I(34)];
	B[MY_I_A(253)] = RCT[MY_I(141)] * V[MY_I(54)];
	B[MY_I_A(254)] = RCT[MY_I(141)] * V[MY_I(35)];
	B[MY_I_A(255)] = RCT[MY_I(142)] * V[MY_I(54)];
	B[MY_I_A(256)] = RCT[MY_I(142)] * V[MY_I(44)];
	B[MY_I_A(257)] = RCT[MY_I(143)] * V[MY_I(50)];
	B[MY_I_A(258)] = RCT[MY_I(143)] * V[MY_I(38)];
	B[MY_I_A(259)] = RCT[MY_I(144)] * V[MY_I(51)];
	B[MY_I_A(260)] = RCT[MY_I(144)] * V[MY_I(38)];
	B[MY_I_A(261)] = RCT[MY_I(145)] * V[MY_I(54)];
	B[MY_I_A(262)] = RCT[MY_I(145)] * V[MY_I(38)];
	B[MY_I_A(263)] = RCT[MY_I(146)] * 2* V [MY_I(38)];
	B[MY_I_A(264)] = RCT[MY_I(147)] * V[MY_I(57)];
	B[MY_I_A(265)] = RCT[MY_I(147)] * V[MY_I(38)];
	B[MY_I_A(266)] = RCT[MY_I(148)] * V[MY_I(58)];
	B[MY_I_A(267)] = RCT[MY_I(148)] * V[MY_I(31)];
	B[MY_I_A(268)] = RCT[MY_I(149)] * V[MY_I(50)];
	B[MY_I_A(269)] = RCT[MY_I(149)] * V[MY_I(31)];
	B[MY_I_A(270)] = RCT[MY_I(150)] * V[MY_I(51)];
	B[MY_I_A(271)] = RCT[MY_I(150)] * V[MY_I(31)];
	B[MY_I_A(272)] = RCT[MY_I(151)] * V[MY_I(54)];
	B[MY_I_A(273)] = RCT[MY_I(151)] * V[MY_I(31)];
	B[MY_I_A(274)] = RCT[MY_I(152)] * 2* V [MY_I(31)];
	B[MY_I_A(275)] = RCT[MY_I(153)] * V[MY_I(51)];
	B[MY_I_A(276)] = RCT[MY_I(153)] * V[MY_I(37)];
	B[MY_I_A(277)] = RCT[MY_I(154)] * V[MY_I(54)];
	B[MY_I_A(278)] = RCT[MY_I(154)] * V[MY_I(37)];
	B[MY_I_A(279)] = RCT[MY_I(155)] * 2* V [MY_I(37)];

	JVS[MY_I(0)] = 0;
	JVS[MY_I(1)] = B[MY_I_A(69)];
	JVS[MY_I(2)] = B[MY_I_A(70)];
	JVS[MY_I(3)] = 0;
	JVS[MY_I(4)] = 0.4*B[MY_I_A(173)];
	JVS[MY_I(5)] = 0.2*B[MY_I_A(179)];
	JVS[MY_I(6)] = 0.2*B[MY_I_A(175)];
	JVS[MY_I(7)] = 0.06*B[MY_I_A(177)];
	JVS[MY_I(8)] = 0.4*B[MY_I_A(174)]+0.2*B[MY_I_A(176)]+0.06*B[MY_I_A(178)]+0.2*B[MY_I_A(180)];
	JVS[MY_I(9)] = 0;
	JVS[MY_I(10)] = 0.2*B[MY_I_A(179)];
	JVS[MY_I(11)] = 0.2*B[MY_I_A(175)];
	JVS[MY_I(12)] = 0.29*B[MY_I_A(177)];
	JVS[MY_I(13)] = 0.5*B[MY_I_A(244)];
	JVS[MY_I(14)] = 0.5*B[MY_I_A(277)];
	JVS[MY_I(15)] = 0.5*B[MY_I_A(242)];
	JVS[MY_I(16)] = 0.5*B[MY_I_A(238)];
	JVS[MY_I(17)] = 0.5*B[MY_I_A(240)];
	JVS[MY_I(18)] = 0.5*B[MY_I_A(232)];
	JVS[MY_I(19)] = 0.2*B[MY_I_A(176)]+0.29*B[MY_I_A(178)]+0.2*B[MY_I_A(180)];
	JVS[MY_I(20)] = 0.5*B[MY_I_A(246)];
	JVS[MY_I(21)] = 0.5*B[MY_I_A(248)];
	JVS[MY_I(22)] = 0.5*B[MY_I_A(226)]+0.5*B[MY_I_A(233)];
	JVS[MY_I(23)] = 0.5*B[MY_I_A(227)]+0.5*B[MY_I_A(234)]+0.5*B[MY_I_A(236)]+0.5*B[MY_I_A(239)]+0.5*B[MY_I_A(241)]+0.5
		   *B[MY_I_A(243)]+0.5*B[MY_I_A(245)]+0.5*B[MY_I_A(247)]+0.5*B[MY_I_A(249)]+0.5*B[MY_I_A(278)];
	JVS[MY_I(24)] = 0.5*B[MY_I_A(237)];
	JVS[MY_I(25)] = 0.5*B[MY_I_A(235)];
	JVS[MY_I(26)] = 0;
	JVS[MY_I(27)] = B[MY_I_A(71)];
	JVS[MY_I(28)] = B[MY_I_A(72)];
	JVS[MY_I(29)] = -B[MY_I_A(69)];
	JVS[MY_I(30)] = -B[MY_I_A(70)];
	JVS[MY_I(31)] = -B[MY_I_A(75)];
	JVS[MY_I(32)] = -B[MY_I_A(76)];
	JVS[MY_I(33)] = -B[MY_I_A(25)]-B[MY_I_A(27)];
	JVS[MY_I(34)] = B[MY_I_A(1)];
	JVS[MY_I(35)] = -B[MY_I_A(79)];
	JVS[MY_I(36)] = -B[MY_I_A(80)];
	JVS[MY_I(37)] = -B[MY_I_A(81)];
	JVS[MY_I(38)] = -B[MY_I_A(82)];
	JVS[MY_I(39)] = -B[MY_I_A(89)];
	JVS[MY_I(40)] = -B[MY_I_A(90)];
	JVS[MY_I(41)] = -B[MY_I_A(91)];
	JVS[MY_I(42)] = -B[MY_I_A(92)];
	JVS[MY_I(43)] = -B[MY_I_A(124)];
	JVS[MY_I(44)] = B[MY_I_A(122)];
	JVS[MY_I(45)] = B[MY_I_A(123)];
	JVS[MY_I(46)] = -B[MY_I_A(3)];
	JVS[MY_I(47)] = B[MY_I_A(45)];
	JVS[MY_I(48)] = B[MY_I_A(46)];
	JVS[MY_I(49)] = -B[MY_I_A(8)]-B[MY_I_A(43)];
	JVS[MY_I(50)] = B[MY_I_A(40)]+B[MY_I_A(41)];
	JVS[MY_I(51)] = -B[MY_I_A(44)];
	JVS[MY_I(52)] = -B[MY_I_A(59)]-B[MY_I_A(60)];
	JVS[MY_I(53)] = B[MY_I_A(57)];
	JVS[MY_I(54)] = B[MY_I_A(58)];
	JVS[MY_I(55)] = -B[MY_I_A(77)];
	JVS[MY_I(56)] = -B[MY_I_A(78)];
	JVS[MY_I(57)] = -B[MY_I_A(73)];
	JVS[MY_I(58)] = 0.06*B[MY_I_A(175)];
	JVS[MY_I(59)] = 0.09*B[MY_I_A(177)];
	JVS[MY_I(60)] = 0.06*B[MY_I_A(176)]+0.09*B[MY_I_A(178)];
	JVS[MY_I(61)] = -B[MY_I_A(74)];
	JVS[MY_I(62)] = -B[MY_I_A(14)]-B[MY_I_A(111)];
	JVS[MY_I(63)] = B[MY_I_A(199)];
	JVS[MY_I(64)] = -B[MY_I_A(112)];
	JVS[MY_I(65)] = B[MY_I_A(200)];
	JVS[MY_I(66)] = B[MY_I_A(25)];
	JVS[MY_I(67)] = -B[MY_I_A(21)]-B[MY_I_A(23)];
	JVS[MY_I(68)] = B[MY_I_A(2)];
	JVS[MY_I(69)] = B[MY_I_A(7)];
	JVS[MY_I(70)] = B[MY_I_A(0)]-B[MY_I_A(24)];
	JVS[MY_I(71)] = -B[MY_I_A(5)]-B[MY_I_A(39)]-B[MY_I_A(65)];
	JVS[MY_I(72)] = B[MY_I_A(37)];
	JVS[MY_I(73)] = -B[MY_I_A(66)];
	JVS[MY_I(74)] = B[MY_I_A(38)];
	JVS[MY_I(75)] = -B[MY_I_A(12)]-B[MY_I_A(107)];
	JVS[MY_I(76)] = B[MY_I_A(181)];
	JVS[MY_I(77)] = B[MY_I_A(182)];
	JVS[MY_I(78)] = -B[MY_I_A(108)];
	JVS[MY_I(79)] = 0.25*B[MY_I_A(89)];
	JVS[MY_I(80)] = 0.17*B[MY_I_A(91)];
	JVS[MY_I(81)] = -B[MY_I_A(93)]-0.5*B[MY_I_A(163)];
	JVS[MY_I(82)] = 0.25*B[MY_I_A(90)]+0.17*B[MY_I_A(92)]-B[MY_I_A(94)];
	JVS[MY_I(83)] = -0.5*B[MY_I_A(164)];
	JVS[MY_I(84)] = -B[MY_I_A(113)]-B[MY_I_A(121)];
	JVS[MY_I(85)] = -B[MY_I_A(114)];
	JVS[MY_I(86)] = B[MY_I_A(119)];
	JVS[MY_I(87)] = B[MY_I_A(120)];
	JVS[MY_I(88)] = -B[MY_I_A(83)]-B[MY_I_A(165)]-B[MY_I_A(173)];
	JVS[MY_I(89)] = -B[MY_I_A(174)];
	JVS[MY_I(90)] = -B[MY_I_A(84)];
	JVS[MY_I(91)] = -B[MY_I_A(166)];
	JVS[MY_I(92)] = 2*B[MY_I_A(60)];
	JVS[MY_I(93)] = B[MY_I_A(163)];
	JVS[MY_I(94)] = -B[MY_I_A(4)]-B[MY_I_A(63)];
	JVS[MY_I(95)] = B[MY_I_A(161)];
	JVS[MY_I(96)] = B[MY_I_A(157)];
	JVS[MY_I(97)] = B[MY_I_A(159)];
	JVS[MY_I(98)] = B[MY_I_A(153)];
	JVS[MY_I(99)] = B[MY_I_A(155)];
	JVS[MY_I(100)] = B[MY_I_A(55)];
	JVS[MY_I(101)] = B[MY_I_A(61)]-B[MY_I_A(64)];
	JVS[MY_I(102)] = B[MY_I_A(56)]+B[MY_I_A(154)]+B[MY_I_A(156)]+B[MY_I_A(158)]+B[MY_I_A(160)]+B[MY_I_A(162)]+B[MY_I_A(164)];
	JVS[MY_I(103)] = B[MY_I_A(62)];
	JVS[MY_I(104)] = 0.42*B[MY_I_A(173)];
	JVS[MY_I(105)] = -B[MY_I_A(71)];
	JVS[MY_I(106)] = 0.33*B[MY_I_A(179)];
	JVS[MY_I(107)] = 0.33*B[MY_I_A(175)];
	JVS[MY_I(108)] = 0.23*B[MY_I_A(177)];
	JVS[MY_I(109)] = 1.87*B[MY_I_A(16)]+1.55*B[MY_I_A(17)]+2*B[MY_I_A(101)]+2*B[MY_I_A(157)];
	JVS[MY_I(110)] = B[MY_I_A(18)]+B[MY_I_A(103)]+B[MY_I_A(159)];
	JVS[MY_I(111)] = B[MY_I_A(9)]+B[MY_I_A(10)]+B[MY_I_A(95)]+B[MY_I_A(153)];
	JVS[MY_I(112)] = 0.95*B[MY_I_A(141)]+0.475*B[MY_I_A(232)]+0.95*B[MY_I_A(255)];
	JVS[MY_I(113)] = 0.42*B[MY_I_A(174)]+0.33*B[MY_I_A(176)]+0.23*B[MY_I_A(178)]+0.33*B[MY_I_A(180)];
	JVS[MY_I(114)] = B[MY_I_A(11)];
	JVS[MY_I(115)] = 0.475*B[MY_I_A(233)];
	JVS[MY_I(116)] = -B[MY_I_A(72)]+B[MY_I_A(96)]+2*B[MY_I_A(102)]+B[MY_I_A(104)];
	JVS[MY_I(117)] = B[MY_I_A(154)]+2*B[MY_I_A(158)]+B[MY_I_A(160)];
	JVS[MY_I(118)] = 0.95*B[MY_I_A(256)];
	JVS[MY_I(119)] = 0.95*B[MY_I_A(142)];
	JVS[MY_I(120)] = -B[MY_I_A(117)]-B[MY_I_A(171)]-B[MY_I_A(179)];
	JVS[MY_I(121)] = -B[MY_I_A(180)];
	JVS[MY_I(122)] = -B[MY_I_A(118)];
	JVS[MY_I(123)] = -B[MY_I_A(172)];
	JVS[MY_I(124)] = -B[MY_I_A(85)]-B[MY_I_A(167)]-B[MY_I_A(175)];
	JVS[MY_I(125)] = -B[MY_I_A(176)];
	JVS[MY_I(126)] = -B[MY_I_A(86)];
	JVS[MY_I(127)] = -B[MY_I_A(168)];
	JVS[MY_I(128)] = -B[MY_I_A(87)]-B[MY_I_A(169)]-B[MY_I_A(177)];
	JVS[MY_I(129)] = -B[MY_I_A(178)];
	JVS[MY_I(130)] = -B[MY_I_A(88)];
	JVS[MY_I(131)] = -B[MY_I_A(170)];
	JVS[MY_I(132)] = -B[MY_I_A(19)]-B[MY_I_A(105)]-B[MY_I_A(161)];
	JVS[MY_I(133)] = 0.7*B[MY_I_A(143)]+0.7*B[MY_I_A(228)]+B[MY_I_A(251)];
	JVS[MY_I(134)] = 0.806*B[MY_I_A(145)]+0.806*B[MY_I_A(230)]+B[MY_I_A(253)];
	JVS[MY_I(135)] = 0.7*B[MY_I_A(229)]+0.806*B[MY_I_A(231)];
	JVS[MY_I(136)] = -B[MY_I_A(106)];
	JVS[MY_I(137)] = -B[MY_I_A(162)];
	JVS[MY_I(138)] = B[MY_I_A(252)]+B[MY_I_A(254)];
	JVS[MY_I(139)] = 0.7*B[MY_I_A(144)]+0.806*B[MY_I_A(146)];
	JVS[MY_I(140)] = -B[MY_I_A(16)]-B[MY_I_A(17)]-B[MY_I_A(101)]-B[MY_I_A(157)];
	JVS[MY_I(141)] = 0.16*B[MY_I_A(143)]+0.16*B[MY_I_A(228)]+0.2*B[MY_I_A(251)];
	JVS[MY_I(142)] = 0.89*B[MY_I_A(141)]+0.445*B[MY_I_A(232)]+0.89*B[MY_I_A(255)];
	JVS[MY_I(143)] = 0.16*B[MY_I_A(229)]+0.445*B[MY_I_A(233)];
	JVS[MY_I(144)] = -B[MY_I_A(102)];
	JVS[MY_I(145)] = -B[MY_I_A(158)];
	JVS[MY_I(146)] = 0.2*B[MY_I_A(252)]+0.89*B[MY_I_A(256)];
	JVS[MY_I(147)] = 0.89*B[MY_I_A(142)]+0.16*B[MY_I_A(144)];
	JVS[MY_I(148)] = B[MY_I_A(163)];
	JVS[MY_I(149)] = -B[MY_I_A(266)]-B[MY_I_A(268)]-B[MY_I_A(270)]-B[MY_I_A(272)]-2*B[MY_I_A(274)];
	JVS[MY_I(150)] = -B[MY_I_A(269)];
	JVS[MY_I(151)] = -B[MY_I_A(271)];
	JVS[MY_I(152)] = 0;
	JVS[MY_I(153)] = B[MY_I_A(164)];
	JVS[MY_I(154)] = -B[MY_I_A(273)];
	JVS[MY_I(155)] = -B[MY_I_A(267)];
	JVS[MY_I(156)] = 0.025*B[MY_I_A(77)];
	JVS[MY_I(157)] = 0.1*B[MY_I_A(177)];
	JVS[MY_I(158)] = -B[MY_I_A(15)]-B[MY_I_A(99)];
	JVS[MY_I(159)] = 0.69*B[MY_I_A(129)]+0.75*B[MY_I_A(214)]+0.86*B[MY_I_A(238)];
	JVS[MY_I(160)] = 1.06*B[MY_I_A(131)]+1.39*B[MY_I_A(216)]+0.9*B[MY_I_A(240)];
	JVS[MY_I(161)] = 0.1*B[MY_I_A(178)];
	JVS[MY_I(162)] = 0.8*B[MY_I_A(20)];
	JVS[MY_I(163)] = 0.1*B[MY_I_A(137)]+0.55*B[MY_I_A(222)]+0.55*B[MY_I_A(246)];
	JVS[MY_I(164)] = 0.6*B[MY_I_A(212)]+0.75*B[MY_I_A(215)]+1.39*B[MY_I_A(217)]+0.55*B[MY_I_A(223)];
	JVS[MY_I(165)] = 0.025*B[MY_I_A(78)]-B[MY_I_A(100)];
	JVS[MY_I(166)] = 0;
	JVS[MY_I(167)] = 0.8*B[MY_I_A(236)]+0.86*B[MY_I_A(239)]+0.9*B[MY_I_A(241)]+0.55*B[MY_I_A(247)];
	JVS[MY_I(168)] = 0.25*B[MY_I_A(127)]+0.6*B[MY_I_A(213)]+0.8*B[MY_I_A(237)];
	JVS[MY_I(169)] = 0.25*B[MY_I_A(128)]+0.69*B[MY_I_A(130)]+1.06*B[MY_I_A(132)]+0.1*B[MY_I_A(138)];
	JVS[MY_I(170)] = -B[MY_I_A(18)]-B[MY_I_A(103)]-B[MY_I_A(159)];
	JVS[MY_I(171)] = 0.17*B[MY_I_A(143)]+0.17*B[MY_I_A(228)]+0.8*B[MY_I_A(251)];
	JVS[MY_I(172)] = 0.45*B[MY_I_A(145)]+0.45*B[MY_I_A(230)]+B[MY_I_A(253)];
	JVS[MY_I(173)] = 0.11*B[MY_I_A(141)]+0.055*B[MY_I_A(232)]+0.11*B[MY_I_A(255)];
	JVS[MY_I(174)] = B[MY_I_A(149)]+0.75*B[MY_I_A(224)]+B[MY_I_A(248)];
	JVS[MY_I(175)] = 0.75*B[MY_I_A(225)]+0.17*B[MY_I_A(229)]+0.45*B[MY_I_A(231)]+0.055*B[MY_I_A(233)];
	JVS[MY_I(176)] = -B[MY_I_A(104)];
	JVS[MY_I(177)] = -B[MY_I_A(160)];
	JVS[MY_I(178)] = B[MY_I_A(249)]+0.8*B[MY_I_A(252)]+B[MY_I_A(254)]+0.11*B[MY_I_A(256)];
	JVS[MY_I(179)] = 0.11*B[MY_I_A(142)]+0.17*B[MY_I_A(144)]+0.45*B[MY_I_A(146)]+B[MY_I_A(150)];
	JVS[MY_I(180)] = 0.75*B[MY_I_A(89)];
	JVS[MY_I(181)] = -B[MY_I_A(143)]-B[MY_I_A(201)]-B[MY_I_A(228)]-B[MY_I_A(251)];
	JVS[MY_I(182)] = -B[MY_I_A(202)];
	JVS[MY_I(183)] = -B[MY_I_A(229)];
	JVS[MY_I(184)] = 0.75*B[MY_I_A(90)];
	JVS[MY_I(185)] = -B[MY_I_A(252)];
	JVS[MY_I(186)] = -B[MY_I_A(144)];
	JVS[MY_I(187)] = 0.83*B[MY_I_A(91)];
	JVS[MY_I(188)] = -B[MY_I_A(145)]-B[MY_I_A(203)]-B[MY_I_A(230)]-B[MY_I_A(253)];
	JVS[MY_I(189)] = -B[MY_I_A(204)];
	JVS[MY_I(190)] = -B[MY_I_A(231)];
	JVS[MY_I(191)] = 0.83*B[MY_I_A(92)];
	JVS[MY_I(192)] = -B[MY_I_A(254)];
	JVS[MY_I(193)] = -B[MY_I_A(146)];
	JVS[MY_I(194)] = B[MY_I_A(117)];
	JVS[MY_I(195)] = B[MY_I_A(85)];
	JVS[MY_I(196)] = -B[MY_I_A(135)]-B[MY_I_A(193)]-B[MY_I_A(220)]-B[MY_I_A(244)];
	JVS[MY_I(197)] = 0;
	JVS[MY_I(198)] = -B[MY_I_A(194)];
	JVS[MY_I(199)] = -B[MY_I_A(221)];
	JVS[MY_I(200)] = B[MY_I_A(86)]+B[MY_I_A(118)];
	JVS[MY_I(201)] = 0;
	JVS[MY_I(202)] = -B[MY_I_A(245)];
	JVS[MY_I(203)] = -B[MY_I_A(136)];
	JVS[MY_I(204)] = B[MY_I_A(165)];
	JVS[MY_I(205)] = B[MY_I_A(171)];
	JVS[MY_I(206)] = B[MY_I_A(167)];
	JVS[MY_I(207)] = B[MY_I_A(169)];
	JVS[MY_I(208)] = -B[MY_I_A(151)]-B[MY_I_A(207)]-B[MY_I_A(275)]-B[MY_I_A(277)]-2*B[MY_I_A(279)];
	JVS[MY_I(209)] = 0;
	JVS[MY_I(210)] = -B[MY_I_A(208)];
	JVS[MY_I(211)] = -B[MY_I_A(276)];
	JVS[MY_I(212)] = 0;
	JVS[MY_I(213)] = B[MY_I_A(166)]+B[MY_I_A(168)]+B[MY_I_A(170)]+B[MY_I_A(172)];
	JVS[MY_I(214)] = -B[MY_I_A(278)];
	JVS[MY_I(215)] = -B[MY_I_A(152)];
	JVS[MY_I(216)] = 0.25*B[MY_I_A(79)];
	JVS[MY_I(217)] = 0.75*B[MY_I_A(81)];
	JVS[MY_I(218)] = 0.9*B[MY_I_A(93)];
	JVS[MY_I(219)] = B[MY_I_A(113)];
	JVS[MY_I(220)] = -B[MY_I_A(257)]-B[MY_I_A(259)]-B[MY_I_A(261)]-2*B[MY_I_A(263)]-B[MY_I_A(264)];
	JVS[MY_I(221)] = 2*B[MY_I_A(141)]+B[MY_I_A(232)]+2*B[MY_I_A(255)];
	JVS[MY_I(222)] = -B[MY_I_A(258)];
	JVS[MY_I(223)] = B[MY_I_A(233)]-B[MY_I_A(260)];
	JVS[MY_I(224)] = 0.25*B[MY_I_A(80)]+0.75*B[MY_I_A(82)]+0.9*B[MY_I_A(94)]+B[MY_I_A(114)];
	JVS[MY_I(225)] = 0;
	JVS[MY_I(226)] = 2*B[MY_I_A(256)]-B[MY_I_A(262)];
	JVS[MY_I(227)] = 2*B[MY_I_A(142)]-B[MY_I_A(265)];
	JVS[MY_I(228)] = 0;
	JVS[MY_I(229)] = B[MY_I_A(83)];
	JVS[MY_I(230)] = -B[MY_I_A(133)]-B[MY_I_A(191)]-B[MY_I_A(218)]-B[MY_I_A(242)];
	JVS[MY_I(231)] = 0;
	JVS[MY_I(232)] = -B[MY_I_A(192)];
	JVS[MY_I(233)] = -B[MY_I_A(219)];
	JVS[MY_I(234)] = B[MY_I_A(84)];
	JVS[MY_I(235)] = 0;
	JVS[MY_I(236)] = -B[MY_I_A(243)];
	JVS[MY_I(237)] = -B[MY_I_A(134)];
	JVS[MY_I(238)] = B[MY_I_A(79)];
	JVS[MY_I(239)] = -B[MY_I_A(129)]-B[MY_I_A(187)]-B[MY_I_A(214)]-B[MY_I_A(238)];
	JVS[MY_I(240)] = -B[MY_I_A(188)];
	JVS[MY_I(241)] = -B[MY_I_A(215)];
	JVS[MY_I(242)] = B[MY_I_A(80)];
	JVS[MY_I(243)] = -B[MY_I_A(239)];
	JVS[MY_I(244)] = -B[MY_I_A(130)];
	JVS[MY_I(245)] = B[MY_I_A(268)];
	JVS[MY_I(246)] = B[MY_I_A(201)];
	JVS[MY_I(247)] = B[MY_I_A(203)];
	JVS[MY_I(248)] = B[MY_I_A(193)];
	JVS[MY_I(249)] = B[MY_I_A(257)];
	JVS[MY_I(250)] = B[MY_I_A(191)];
	JVS[MY_I(251)] = B[MY_I_A(187)];
	JVS[MY_I(252)] = -B[MY_I_A(13)]-B[MY_I_A(109)];
	JVS[MY_I(253)] = B[MY_I_A(189)];
	JVS[MY_I(254)] = B[MY_I_A(205)];
	JVS[MY_I(255)] = 0;
	JVS[MY_I(256)] = B[MY_I_A(195)];
	JVS[MY_I(257)] = B[MY_I_A(197)];
	JVS[MY_I(258)] = B[MY_I_A(183)]+B[MY_I_A(185)]+B[MY_I_A(188)]+B[MY_I_A(190)]+B[MY_I_A(192)]+B[MY_I_A(194)]+B[MY_I_A(196)]+B[MY_I_A(198)]
			+B[MY_I_A(202)]+B[MY_I_A(204)]+B[MY_I_A(206)]+B[MY_I_A(258)]+B[MY_I_A(269)];
	JVS[MY_I(259)] = 0;
	JVS[MY_I(260)] = -B[MY_I_A(110)];
	JVS[MY_I(261)] = 0;
	JVS[MY_I(262)] = 0;
	JVS[MY_I(263)] = B[MY_I_A(186)];
	JVS[MY_I(264)] = B[MY_I_A(184)];
	JVS[MY_I(265)] = 0;
	JVS[MY_I(266)] = 0;
	JVS[MY_I(267)] = 0.009*B[MY_I_A(77)];
	JVS[MY_I(268)] = B[MY_I_A(12)]+0.5*B[MY_I_A(107)];
	JVS[MY_I(269)] = B[MY_I_A(113)];
	JVS[MY_I(270)] = B[MY_I_A(173)];
	JVS[MY_I(271)] = 0.53*B[MY_I_A(179)];
	JVS[MY_I(272)] = 0.53*B[MY_I_A(175)];
	JVS[MY_I(273)] = 0.18*B[MY_I_A(177)];
	JVS[MY_I(274)] = 0.13*B[MY_I_A(16)]+0.45*B[MY_I_A(17)];
	JVS[MY_I(275)] = B[MY_I_A(270)];
	JVS[MY_I(276)] = B[MY_I_A(228)];
	JVS[MY_I(277)] = B[MY_I_A(230)];
	JVS[MY_I(278)] = B[MY_I_A(135)]+1.25*B[MY_I_A(220)]+0.5*B[MY_I_A(244)];
	JVS[MY_I(279)] = B[MY_I_A(151)]+1.75*B[MY_I_A(275)]+B[MY_I_A(277)]+2*B[MY_I_A(279)];
	JVS[MY_I(280)] = B[MY_I_A(259)];
	JVS[MY_I(281)] = 1.6*B[MY_I_A(133)]+1.55*B[MY_I_A(218)]+0.8*B[MY_I_A(242)];
	JVS[MY_I(282)] = 0.77*B[MY_I_A(214)];
	JVS[MY_I(283)] = -B[MY_I_A(9)]-B[MY_I_A(10)]-B[MY_I_A(95)]-B[MY_I_A(153)];
	JVS[MY_I(284)] = 0.04*B[MY_I_A(131)]+0.8*B[MY_I_A(216)];
	JVS[MY_I(285)] = 0.5*B[MY_I_A(232)];
	JVS[MY_I(286)] = B[MY_I_A(174)]+0.53*B[MY_I_A(176)]+0.18*B[MY_I_A(178)]+0.53*B[MY_I_A(180)];
	JVS[MY_I(287)] = 0.28*B[MY_I_A(137)]+0.89*B[MY_I_A(222)]+0.14*B[MY_I_A(246)];
	JVS[MY_I(288)] = 0.75*B[MY_I_A(224)];
	JVS[MY_I(289)] = 0;
	JVS[MY_I(290)] = B[MY_I_A(125)]+1.5*B[MY_I_A(209)]+0.75*B[MY_I_A(210)]+0.75*B[MY_I_A(212)]+0.77*B[MY_I_A(215)]+0.8
			*B[MY_I_A(217)]+1.55*B[MY_I_A(219)]+1.25*B[MY_I_A(221)]+0.89*B[MY_I_A(223)]+0.75*B[MY_I_A(225)]
			+B[MY_I_A(226)]+B[MY_I_A(229)]+B[MY_I_A(231)]+0.5*B[MY_I_A(233)]+B[MY_I_A(260)]+B[MY_I_A(271)]+1.75*B[MY_I_A(276)];
	JVS[MY_I(291)] = 0.009*B[MY_I_A(78)]-B[MY_I_A(96)]+0.5*B[MY_I_A(108)]+B[MY_I_A(114)];
	JVS[MY_I(292)] = -B[MY_I_A(154)];
	JVS[MY_I(293)] = B[MY_I_A(227)]+0.8*B[MY_I_A(243)]+0.5*B[MY_I_A(245)]+0.14*B[MY_I_A(247)]+B[MY_I_A(278)];
	JVS[MY_I(294)] = 0.09*B[MY_I_A(127)]+0.75*B[MY_I_A(213)];
	JVS[MY_I(295)] = 0.75*B[MY_I_A(211)];
	JVS[MY_I(296)] = B[MY_I_A(126)]+0.09*B[MY_I_A(128)]+0.04*B[MY_I_A(132)]+1.6*B[MY_I_A(134)]+B[MY_I_A(136)]+0.28
			*B[MY_I_A(138)]+B[MY_I_A(152)];
	JVS[MY_I(297)] = 0;
	JVS[MY_I(298)] = B[MY_I_A(81)];
	JVS[MY_I(299)] = -B[MY_I_A(131)]-B[MY_I_A(189)]-B[MY_I_A(216)]-B[MY_I_A(240)];
	JVS[MY_I(300)] = -B[MY_I_A(190)];
	JVS[MY_I(301)] = -B[MY_I_A(217)];
	JVS[MY_I(302)] = B[MY_I_A(82)];
	JVS[MY_I(303)] = -B[MY_I_A(241)];
	JVS[MY_I(304)] = -B[MY_I_A(132)];
	JVS[MY_I(305)] = B[MY_I_A(124)];
	JVS[MY_I(306)] = 0.9*B[MY_I_A(93)];
	JVS[MY_I(307)] = B[MY_I_A(19)]+B[MY_I_A(105)]+B[MY_I_A(161)];
	JVS[MY_I(308)] = 0;
	JVS[MY_I(309)] = 0;
	JVS[MY_I(310)] = -B[MY_I_A(122)]-B[MY_I_A(141)]-B[MY_I_A(205)]-B[MY_I_A(232)]-B[MY_I_A(255)];
	JVS[MY_I(311)] = -B[MY_I_A(206)];
	JVS[MY_I(312)] = -B[MY_I_A(233)];
	JVS[MY_I(313)] = 0.9*B[MY_I_A(94)]+B[MY_I_A(106)];
	JVS[MY_I(314)] = B[MY_I_A(162)];
	JVS[MY_I(315)] = -B[MY_I_A(256)];
	JVS[MY_I(316)] = -B[MY_I_A(142)];
	JVS[MY_I(317)] = -B[MY_I_A(123)];
	JVS[MY_I(318)] = B[MY_I_A(21)];
	JVS[MY_I(319)] = -B[MY_I_A(173)];
	JVS[MY_I(320)] = -B[MY_I_A(179)];
	JVS[MY_I(321)] = -B[MY_I_A(175)];
	JVS[MY_I(322)] = -B[MY_I_A(177)];
	JVS[MY_I(323)] = -B[MY_I_A(1)]-B[MY_I_A(2)]-B[MY_I_A(29)]-B[MY_I_A(31)]-B[MY_I_A(33)]-B[MY_I_A(49)]-B[MY_I_A(174)]-B[MY_I_A(176)]-B[MY_I_A(178)]
			-B[MY_I_A(180)];
	JVS[MY_I(324)] = -B[MY_I_A(34)];
	JVS[MY_I(325)] = -B[MY_I_A(32)];
	JVS[MY_I(326)] = 0;
	JVS[MY_I(327)] = -B[MY_I_A(30)];
	JVS[MY_I(328)] = -B[MY_I_A(50)];
	JVS[MY_I(329)] = B[MY_I_A(266)];
	JVS[MY_I(330)] = B[MY_I_A(207)];
	JVS[MY_I(331)] = 0.08*B[MY_I_A(129)];
	JVS[MY_I(332)] = 0.24*B[MY_I_A(131)];
	JVS[MY_I(333)] = 0;
	JVS[MY_I(334)] = -B[MY_I_A(20)]-B[MY_I_A(115)];
	JVS[MY_I(335)] = B[MY_I_A(208)];
	JVS[MY_I(336)] = 0;
	JVS[MY_I(337)] = -B[MY_I_A(116)];
	JVS[MY_I(338)] = 0;
	JVS[MY_I(339)] = 0;
	JVS[MY_I(340)] = 0.036*B[MY_I_A(127)];
	JVS[MY_I(341)] = 0.036*B[MY_I_A(128)]+0.08*B[MY_I_A(130)]+0.24*B[MY_I_A(132)];
	JVS[MY_I(342)] = B[MY_I_A(267)];
	JVS[MY_I(343)] = 0.075*B[MY_I_A(77)];
	JVS[MY_I(344)] = 0.5*B[MY_I_A(179)];
	JVS[MY_I(345)] = 0.5*B[MY_I_A(175)];
	JVS[MY_I(346)] = 0.72*B[MY_I_A(177)];
	JVS[MY_I(347)] = B[MY_I_A(135)]+0.75*B[MY_I_A(220)]+B[MY_I_A(244)];
	JVS[MY_I(348)] = B[MY_I_A(151)]+B[MY_I_A(275)]+B[MY_I_A(277)]+2*B[MY_I_A(279)];
	JVS[MY_I(349)] = 0.2*B[MY_I_A(133)]+0.35*B[MY_I_A(218)]+0.6*B[MY_I_A(242)];
	JVS[MY_I(350)] = 0.38*B[MY_I_A(129)]+0.41*B[MY_I_A(214)]+0.14*B[MY_I_A(238)];
	JVS[MY_I(351)] = B[MY_I_A(13)]+0.5*B[MY_I_A(109)];
	JVS[MY_I(352)] = 0.35*B[MY_I_A(131)]+0.46*B[MY_I_A(216)]+0.1*B[MY_I_A(240)];
	JVS[MY_I(353)] = 0;
	JVS[MY_I(354)] = 0.5*B[MY_I_A(176)]+0.72*B[MY_I_A(178)]+0.5*B[MY_I_A(180)];
	JVS[MY_I(355)] = 0.2*B[MY_I_A(20)];
	JVS[MY_I(356)] = -B[MY_I_A(11)]-B[MY_I_A(97)]-B[MY_I_A(155)];
	JVS[MY_I(357)] = 1.45*B[MY_I_A(137)]+0.725*B[MY_I_A(222)]+0.725*B[MY_I_A(246)];
	JVS[MY_I(358)] = 0;
	JVS[MY_I(359)] = 0;
	JVS[MY_I(360)] = 0.75*B[MY_I_A(210)]+0.15*B[MY_I_A(212)]+0.41*B[MY_I_A(215)]+0.46*B[MY_I_A(217)]+0.35
			*B[MY_I_A(219)]+0.75*B[MY_I_A(221)]+0.725*B[MY_I_A(223)]+B[MY_I_A(276)];
	JVS[MY_I(361)] = 0.075*B[MY_I_A(78)]-B[MY_I_A(98)]+0.5*B[MY_I_A(110)];
	JVS[MY_I(362)] = -B[MY_I_A(156)];
	JVS[MY_I(363)] = B[MY_I_A(234)]+0.2*B[MY_I_A(236)]+0.14*B[MY_I_A(239)]+0.1*B[MY_I_A(241)]+0.6*B[MY_I_A(243)]+B[MY_I_A(245)]
			+0.725*B[MY_I_A(247)]+B[MY_I_A(278)];
	JVS[MY_I(364)] = 0.75*B[MY_I_A(127)]+0.15*B[MY_I_A(213)]+0.2*B[MY_I_A(237)];
	JVS[MY_I(365)] = B[MY_I_A(147)]+0.75*B[MY_I_A(211)]+B[MY_I_A(235)];
	JVS[MY_I(366)] = 0.75*B[MY_I_A(128)]+0.38*B[MY_I_A(130)]+0.35*B[MY_I_A(132)]+0.2*B[MY_I_A(134)]+B[MY_I_A(136)]+1.45
			*B[MY_I_A(138)]+B[MY_I_A(148)]+B[MY_I_A(152)];
	JVS[MY_I(367)] = 0;
	JVS[MY_I(368)] = B[MY_I_A(87)];
	JVS[MY_I(369)] = 0;
	JVS[MY_I(370)] = -B[MY_I_A(137)]-B[MY_I_A(195)]-B[MY_I_A(222)]-B[MY_I_A(246)];
	JVS[MY_I(371)] = -B[MY_I_A(196)];
	JVS[MY_I(372)] = -B[MY_I_A(223)];
	JVS[MY_I(373)] = B[MY_I_A(88)];
	JVS[MY_I(374)] = 0;
	JVS[MY_I(375)] = -B[MY_I_A(247)];
	JVS[MY_I(376)] = -B[MY_I_A(138)];
	JVS[MY_I(377)] = 0;
	JVS[MY_I(378)] = B[MY_I_A(99)];
	JVS[MY_I(379)] = 0;
	JVS[MY_I(380)] = 0;
	JVS[MY_I(381)] = 0;
	JVS[MY_I(382)] = 0;
	JVS[MY_I(383)] = 0;
	JVS[MY_I(384)] = -B[MY_I_A(149)]-B[MY_I_A(197)]-B[MY_I_A(224)]-B[MY_I_A(248)];
	JVS[MY_I(385)] = -B[MY_I_A(198)];
	JVS[MY_I(386)] = -B[MY_I_A(225)];
	JVS[MY_I(387)] = B[MY_I_A(100)];
	JVS[MY_I(388)] = 0;
	JVS[MY_I(389)] = -B[MY_I_A(249)];
	JVS[MY_I(390)] = 0;
	JVS[MY_I(391)] = -B[MY_I_A(150)];
	JVS[MY_I(392)] = 0;
	JVS[MY_I(393)] = B[MY_I_A(69)];
	JVS[MY_I(394)] = 0.25*B[MY_I_A(89)];
	JVS[MY_I(395)] = 0.17*B[MY_I_A(91)];
	JVS[MY_I(396)] = B[MY_I_A(43)];
	JVS[MY_I(397)] = 0.17*B[MY_I_A(77)];
	JVS[MY_I(398)] = 0.65*B[MY_I_A(5)]+B[MY_I_A(39)];
	JVS[MY_I(399)] = B[MY_I_A(12)];
	JVS[MY_I(400)] = 0.1*B[MY_I_A(93)];
	JVS[MY_I(401)] = 0.12*B[MY_I_A(173)];
	JVS[MY_I(402)] = B[MY_I_A(71)];
	JVS[MY_I(403)] = 0.23*B[MY_I_A(179)];
	JVS[MY_I(404)] = 0.23*B[MY_I_A(175)];
	JVS[MY_I(405)] = 0.26*B[MY_I_A(177)];
	JVS[MY_I(406)] = B[MY_I_A(19)];
	JVS[MY_I(407)] = 0.8*B[MY_I_A(17)]+B[MY_I_A(101)]+B[MY_I_A(157)];
	JVS[MY_I(408)] = -B[MY_I_A(268)]+B[MY_I_A(270)];
	JVS[MY_I(409)] = B[MY_I_A(18)];
	JVS[MY_I(410)] = B[MY_I_A(143)]-B[MY_I_A(201)]+2*B[MY_I_A(228)]+B[MY_I_A(251)];
	JVS[MY_I(411)] = B[MY_I_A(145)]-B[MY_I_A(203)]+2*B[MY_I_A(230)]+B[MY_I_A(253)];
	JVS[MY_I(412)] = B[MY_I_A(135)]-B[MY_I_A(193)]+B[MY_I_A(220)]+0.5*B[MY_I_A(244)];
	JVS[MY_I(413)] = -B[MY_I_A(207)]+0.5*B[MY_I_A(275)];
	JVS[MY_I(414)] = -B[MY_I_A(257)]+B[MY_I_A(259)];
	JVS[MY_I(415)] = B[MY_I_A(133)]-B[MY_I_A(191)]+B[MY_I_A(218)]+0.5*B[MY_I_A(242)];
	JVS[MY_I(416)] = 0.92*B[MY_I_A(129)]-B[MY_I_A(187)]+B[MY_I_A(214)]+0.5*B[MY_I_A(238)];
	JVS[MY_I(417)] = B[MY_I_A(13)];
	JVS[MY_I(418)] = 2*B[MY_I_A(10)]+B[MY_I_A(95)]+B[MY_I_A(153)];
	JVS[MY_I(419)] = 0.76*B[MY_I_A(131)]-B[MY_I_A(189)]+B[MY_I_A(216)]+0.5*B[MY_I_A(240)];
	JVS[MY_I(420)] = 0.92*B[MY_I_A(141)]-B[MY_I_A(205)]+0.46*B[MY_I_A(232)]+0.92*B[MY_I_A(255)];
	JVS[MY_I(421)] = B[MY_I_A(31)]-B[MY_I_A(33)]+0.12*B[MY_I_A(174)]+0.23*B[MY_I_A(176)]+0.26*B[MY_I_A(178)]+0.23
			*B[MY_I_A(180)];
	JVS[MY_I(422)] = B[MY_I_A(20)];
	JVS[MY_I(423)] = B[MY_I_A(11)];
	JVS[MY_I(424)] = B[MY_I_A(137)]-B[MY_I_A(195)]+B[MY_I_A(222)]+0.5*B[MY_I_A(246)];
	JVS[MY_I(425)] = B[MY_I_A(149)]-B[MY_I_A(197)]+B[MY_I_A(224)]+0.5*B[MY_I_A(248)];
	JVS[MY_I(426)] = -B[MY_I_A(34)]-B[MY_I_A(35)]-B[MY_I_A(37)]-2*B[MY_I_A(40)]-2*B[MY_I_A(41)]-B[MY_I_A(55)]-B[MY_I_A(67)]-B[MY_I_A(181)]
			-B[MY_I_A(183)]-B[MY_I_A(185)]-B[MY_I_A(188)]-B[MY_I_A(190)]-B[MY_I_A(192)]-B[MY_I_A(194)]-B[MY_I_A(196)]-B[MY_I_A(198)]
			-B[MY_I_A(199)]-B[MY_I_A(202)]-B[MY_I_A(204)]-B[MY_I_A(206)]-B[MY_I_A(208)]-B[MY_I_A(258)]-B[MY_I_A(269)];
	JVS[MY_I(427)] = B[MY_I_A(125)]-B[MY_I_A(182)]+B[MY_I_A(209)]+B[MY_I_A(210)]+B[MY_I_A(212)]+B[MY_I_A(215)]+B[MY_I_A(217)]+B[MY_I_A(219)]
			+B[MY_I_A(221)]+B[MY_I_A(223)]+B[MY_I_A(225)]+0.5*B[MY_I_A(226)]+2*B[MY_I_A(229)]+2*B[MY_I_A(231)]+0.46
			*B[MY_I_A(233)]+B[MY_I_A(260)]+B[MY_I_A(271)]+0.5*B[MY_I_A(276)];
	JVS[MY_I(428)] = B[MY_I_A(32)]+B[MY_I_A(44)]-B[MY_I_A(68)]+B[MY_I_A(70)]+B[MY_I_A(72)]+0.17*B[MY_I_A(78)]+0.25*B[MY_I_A(90)]+0.17
			*B[MY_I_A(92)]+0.1*B[MY_I_A(94)]+B[MY_I_A(96)]+B[MY_I_A(102)];
	JVS[MY_I(429)] = -B[MY_I_A(56)]+B[MY_I_A(154)]+B[MY_I_A(158)];
	JVS[MY_I(430)] = -B[MY_I_A(200)]+0.5*B[MY_I_A(227)]+0.5*B[MY_I_A(234)]+0.5*B[MY_I_A(236)]+0.5*B[MY_I_A(239)]+0.5
			*B[MY_I_A(241)]+0.5*B[MY_I_A(243)]+0.5*B[MY_I_A(245)]+0.5*B[MY_I_A(247)]+0.5*B[MY_I_A(249)]+B[MY_I_A(252)]
			+B[MY_I_A(254)]+0.92*B[MY_I_A(256)];
	JVS[MY_I(431)] = 0.964*B[MY_I_A(127)]-B[MY_I_A(186)]+B[MY_I_A(213)]+0.5*B[MY_I_A(237)];
	JVS[MY_I(432)] = B[MY_I_A(147)]-B[MY_I_A(184)]+B[MY_I_A(211)]+0.5*B[MY_I_A(235)];
	JVS[MY_I(433)] = -B[MY_I_A(36)]+B[MY_I_A(126)]+0.964*B[MY_I_A(128)]+0.92*B[MY_I_A(130)]+0.76*B[MY_I_A(132)]+B[MY_I_A(134)]
			+B[MY_I_A(136)]+B[MY_I_A(138)]+0.92*B[MY_I_A(142)]+B[MY_I_A(144)]+B[MY_I_A(146)]+B[MY_I_A(148)]+B[MY_I_A(150)];
	JVS[MY_I(434)] = -B[MY_I_A(38)];
	JVS[MY_I(435)] = B[MY_I_A(73)];
	JVS[MY_I(436)] = B[MY_I_A(14)];
	JVS[MY_I(437)] = 0.5*B[MY_I_A(107)];
	JVS[MY_I(438)] = 0.22*B[MY_I_A(179)];
	JVS[MY_I(439)] = 0.22*B[MY_I_A(175)];
	JVS[MY_I(440)] = 0.31*B[MY_I_A(177)];
	JVS[MY_I(441)] = -B[MY_I_A(270)]+B[MY_I_A(272)];
	JVS[MY_I(442)] = -B[MY_I_A(228)]+B[MY_I_A(251)];
	JVS[MY_I(443)] = -B[MY_I_A(230)]+B[MY_I_A(253)];
	JVS[MY_I(444)] = -B[MY_I_A(220)]+0.5*B[MY_I_A(244)];
	JVS[MY_I(445)] = -B[MY_I_A(275)]+0.5*B[MY_I_A(277)];
	JVS[MY_I(446)] = -B[MY_I_A(259)]+B[MY_I_A(261)];
	JVS[MY_I(447)] = -B[MY_I_A(218)]+0.5*B[MY_I_A(242)];
	JVS[MY_I(448)] = -B[MY_I_A(214)]+0.5*B[MY_I_A(238)];
	JVS[MY_I(449)] = -B[MY_I_A(216)]+0.5*B[MY_I_A(240)];
	JVS[MY_I(450)] = -B[MY_I_A(232)]+B[MY_I_A(255)];
	JVS[MY_I(451)] = 0.22*B[MY_I_A(176)]+0.31*B[MY_I_A(178)]+0.22*B[MY_I_A(180)];
	JVS[MY_I(452)] = B[MY_I_A(11)];
	JVS[MY_I(453)] = -B[MY_I_A(222)]+0.5*B[MY_I_A(246)];
	JVS[MY_I(454)] = -B[MY_I_A(224)]+0.5*B[MY_I_A(248)];
	JVS[MY_I(455)] = -B[MY_I_A(181)];
	JVS[MY_I(456)] = -B[MY_I_A(125)]-B[MY_I_A(182)]-2*B[MY_I_A(209)]-B[MY_I_A(210)]-B[MY_I_A(212)]-B[MY_I_A(215)]-B[MY_I_A(217)]-B[MY_I_A(219)]
			-B[MY_I_A(221)]-B[MY_I_A(223)]-B[MY_I_A(225)]-0.5*B[MY_I_A(226)]-B[MY_I_A(229)]-B[MY_I_A(231)]-B[MY_I_A(233)]
			-B[MY_I_A(260)]-B[MY_I_A(271)]-B[MY_I_A(276)];
	JVS[MY_I(457)] = B[MY_I_A(74)]+0.5*B[MY_I_A(108)];
	JVS[MY_I(458)] = 0;
	JVS[MY_I(459)] = B[MY_I_A(139)]-0.5*B[MY_I_A(227)]+0.5*B[MY_I_A(234)]+0.5*B[MY_I_A(236)]+0.5*B[MY_I_A(239)]+0.5
			*B[MY_I_A(241)]+0.5*B[MY_I_A(243)]+0.5*B[MY_I_A(245)]+0.5*B[MY_I_A(247)]+0.5*B[MY_I_A(249)]+2
			*B[MY_I_A(250)]+B[MY_I_A(252)]+B[MY_I_A(254)]+B[MY_I_A(256)]+B[MY_I_A(262)]+B[MY_I_A(273)]+0.5*B[MY_I_A(278)];
	JVS[MY_I(460)] = -B[MY_I_A(213)]+0.5*B[MY_I_A(237)];
	JVS[MY_I(461)] = -B[MY_I_A(211)]+0.5*B[MY_I_A(235)];
	JVS[MY_I(462)] = -B[MY_I_A(126)]+B[MY_I_A(140)];
	JVS[MY_I(463)] = 0;
	JVS[MY_I(464)] = -B[MY_I_A(69)];
	JVS[MY_I(465)] = -B[MY_I_A(75)];
	JVS[MY_I(466)] = 2*B[MY_I_A(27)];
	JVS[MY_I(467)] = -B[MY_I_A(79)];
	JVS[MY_I(468)] = -B[MY_I_A(81)];
	JVS[MY_I(469)] = -B[MY_I_A(89)];
	JVS[MY_I(470)] = -B[MY_I_A(91)];
	JVS[MY_I(471)] = B[MY_I_A(3)];
	JVS[MY_I(472)] = 2*B[MY_I_A(8)]-B[MY_I_A(43)];
	JVS[MY_I(473)] = -B[MY_I_A(77)];
	JVS[MY_I(474)] = -B[MY_I_A(73)];
	JVS[MY_I(475)] = B[MY_I_A(14)]-B[MY_I_A(111)];
	JVS[MY_I(476)] = 0.35*B[MY_I_A(5)]-B[MY_I_A(65)];
	JVS[MY_I(477)] = B[MY_I_A(12)]-0.5*B[MY_I_A(107)];
	JVS[MY_I(478)] = -1.9*B[MY_I_A(93)];
	JVS[MY_I(479)] = -B[MY_I_A(113)];
	JVS[MY_I(480)] = -B[MY_I_A(83)];
	JVS[MY_I(481)] = B[MY_I_A(4)]-B[MY_I_A(63)];
	JVS[MY_I(482)] = -B[MY_I_A(71)];
	JVS[MY_I(483)] = -B[MY_I_A(117)]+0.1*B[MY_I_A(179)];
	JVS[MY_I(484)] = -B[MY_I_A(85)]+0.1*B[MY_I_A(175)];
	JVS[MY_I(485)] = -B[MY_I_A(87)]+0.14*B[MY_I_A(177)];
	JVS[MY_I(486)] = -B[MY_I_A(105)];
	JVS[MY_I(487)] = -B[MY_I_A(101)];
	JVS[MY_I(488)] = -B[MY_I_A(99)];
	JVS[MY_I(489)] = -B[MY_I_A(103)];
	JVS[MY_I(490)] = 0;
	JVS[MY_I(491)] = 0;
	JVS[MY_I(492)] = 0;
	JVS[MY_I(493)] = B[MY_I_A(13)]-0.5*B[MY_I_A(109)];
	JVS[MY_I(494)] = -B[MY_I_A(95)];
	JVS[MY_I(495)] = 0;
	JVS[MY_I(496)] = 0;
	JVS[MY_I(497)] = -B[MY_I_A(31)]+B[MY_I_A(33)]+0.1*B[MY_I_A(176)]+0.14*B[MY_I_A(178)]+0.1*B[MY_I_A(180)];
	JVS[MY_I(498)] = -B[MY_I_A(115)];
	JVS[MY_I(499)] = -B[MY_I_A(97)];
	JVS[MY_I(500)] = 0;
	JVS[MY_I(501)] = 0;
	JVS[MY_I(502)] = B[MY_I_A(34)]+B[MY_I_A(35)]-B[MY_I_A(67)];
	JVS[MY_I(503)] = 0;
	JVS[MY_I(504)] = -B[MY_I_A(32)]-B[MY_I_A(44)]-B[MY_I_A(45)]-B[MY_I_A(61)]-B[MY_I_A(64)]-B[MY_I_A(66)]-B[MY_I_A(68)]-B[MY_I_A(70)]-B[MY_I_A(72)]
			-B[MY_I_A(74)]-B[MY_I_A(76)]-B[MY_I_A(78)]-B[MY_I_A(80)]-B[MY_I_A(82)]-B[MY_I_A(84)]-B[MY_I_A(86)]-B[MY_I_A(88)]-B[MY_I_A(90)]
			-B[MY_I_A(92)]-1.9*B[MY_I_A(94)]-B[MY_I_A(96)]-B[MY_I_A(98)]-B[MY_I_A(100)]-B[MY_I_A(102)]-B[MY_I_A(104)]-B[MY_I_A(106)]
			-0.5*B[MY_I_A(108)]-0.5*B[MY_I_A(110)]-B[MY_I_A(112)]-B[MY_I_A(114)]-B[MY_I_A(116)]-B[MY_I_A(118)];
	JVS[MY_I(505)] = 0;
	JVS[MY_I(506)] = 0;
	JVS[MY_I(507)] = 0;
	JVS[MY_I(508)] = 0;
	JVS[MY_I(509)] = B[MY_I_A(36)]-B[MY_I_A(46)];
	JVS[MY_I(510)] = -B[MY_I_A(62)];
	JVS[MY_I(511)] = B[MY_I_A(59)];
	JVS[MY_I(512)] = 0.35*B[MY_I_A(5)];
	JVS[MY_I(513)] = -B[MY_I_A(163)];
	JVS[MY_I(514)] = B[MY_I_A(113)];
	JVS[MY_I(515)] = -B[MY_I_A(165)];
	JVS[MY_I(516)] = B[MY_I_A(63)];
	JVS[MY_I(517)] = -B[MY_I_A(171)];
	JVS[MY_I(518)] = -B[MY_I_A(167)];
	JVS[MY_I(519)] = -B[MY_I_A(169)];
	JVS[MY_I(520)] = -B[MY_I_A(161)];
	JVS[MY_I(521)] = -B[MY_I_A(157)];
	JVS[MY_I(522)] = -B[MY_I_A(159)];
	JVS[MY_I(523)] = 0;
	JVS[MY_I(524)] = 0;
	JVS[MY_I(525)] = -B[MY_I_A(153)];
	JVS[MY_I(526)] = 0;
	JVS[MY_I(527)] = 0;
	JVS[MY_I(528)] = B[MY_I_A(49)];
	JVS[MY_I(529)] = -B[MY_I_A(155)];
	JVS[MY_I(530)] = 0;
	JVS[MY_I(531)] = 0;
	JVS[MY_I(532)] = -B[MY_I_A(55)];
	JVS[MY_I(533)] = 0;
	JVS[MY_I(534)] = B[MY_I_A(64)]+B[MY_I_A(114)];
	JVS[MY_I(535)] = -B[MY_I_A(6)]-B[MY_I_A(7)]-B[MY_I_A(51)]-B[MY_I_A(53)]-B[MY_I_A(56)]-B[MY_I_A(57)]-B[MY_I_A(154)]-B[MY_I_A(156)]-B[MY_I_A(158)]
			-B[MY_I_A(160)]-B[MY_I_A(162)]-B[MY_I_A(164)]-B[MY_I_A(166)]-B[MY_I_A(168)]-B[MY_I_A(170)]-B[MY_I_A(172)];
	JVS[MY_I(536)] = 0;
	JVS[MY_I(537)] = 0;
	JVS[MY_I(538)] = 0;
	JVS[MY_I(539)] = -B[MY_I_A(52)];
	JVS[MY_I(540)] = B[MY_I_A(50)]-B[MY_I_A(54)]-B[MY_I_A(58)];
	JVS[MY_I(541)] = B[MY_I_A(111)];
	JVS[MY_I(542)] = B[MY_I_A(121)];
	JVS[MY_I(543)] = -B[MY_I_A(272)];
	JVS[MY_I(544)] = B[MY_I_A(15)];
	JVS[MY_I(545)] = B[MY_I_A(18)]+B[MY_I_A(103)]+B[MY_I_A(159)];
	JVS[MY_I(546)] = -B[MY_I_A(251)];
	JVS[MY_I(547)] = -B[MY_I_A(253)];
	JVS[MY_I(548)] = -B[MY_I_A(244)];
	JVS[MY_I(549)] = -B[MY_I_A(277)];
	JVS[MY_I(550)] = -B[MY_I_A(261)];
	JVS[MY_I(551)] = -B[MY_I_A(242)];
	JVS[MY_I(552)] = -B[MY_I_A(238)];
	JVS[MY_I(553)] = -B[MY_I_A(240)];
	JVS[MY_I(554)] = 0.05*B[MY_I_A(141)]+0.025*B[MY_I_A(232)]-0.95*B[MY_I_A(255)];
	JVS[MY_I(555)] = 0;
	JVS[MY_I(556)] = 0;
	JVS[MY_I(557)] = B[MY_I_A(97)]+B[MY_I_A(155)];
	JVS[MY_I(558)] = -B[MY_I_A(246)];
	JVS[MY_I(559)] = -B[MY_I_A(248)];
	JVS[MY_I(560)] = -B[MY_I_A(199)];
	JVS[MY_I(561)] = -B[MY_I_A(226)]+0.025*B[MY_I_A(233)];
	JVS[MY_I(562)] = B[MY_I_A(98)]+B[MY_I_A(104)]+B[MY_I_A(112)];
	JVS[MY_I(563)] = B[MY_I_A(156)]+B[MY_I_A(160)];
	JVS[MY_I(564)] = -B[MY_I_A(119)]-B[MY_I_A(139)]-B[MY_I_A(200)]-B[MY_I_A(227)]-B[MY_I_A(234)]-B[MY_I_A(236)]-B[MY_I_A(239)]-B[MY_I_A(241)]
			-B[MY_I_A(243)]-B[MY_I_A(245)]-B[MY_I_A(247)]-B[MY_I_A(249)]-2*B[MY_I_A(250)]-B[MY_I_A(252)]-B[MY_I_A(254)]-0.95
			*B[MY_I_A(256)]-B[MY_I_A(262)]-B[MY_I_A(273)]-B[MY_I_A(278)];
	JVS[MY_I(565)] = -B[MY_I_A(237)];
	JVS[MY_I(566)] = -B[MY_I_A(235)];
	JVS[MY_I(567)] = -B[MY_I_A(140)]+0.05*B[MY_I_A(142)];
	JVS[MY_I(568)] = -B[MY_I_A(120)];
	JVS[MY_I(569)] = 0.83*B[MY_I_A(77)];
	JVS[MY_I(570)] = 0.5*B[MY_I_A(109)];
	JVS[MY_I(571)] = 0;
	JVS[MY_I(572)] = 0;
	JVS[MY_I(573)] = 0;
	JVS[MY_I(574)] = B[MY_I_A(115)];
	JVS[MY_I(575)] = 0;
	JVS[MY_I(576)] = 0;
	JVS[MY_I(577)] = -B[MY_I_A(185)];
	JVS[MY_I(578)] = -B[MY_I_A(212)];
	JVS[MY_I(579)] = 0.83*B[MY_I_A(78)]+0.5*B[MY_I_A(110)]+B[MY_I_A(116)];
	JVS[MY_I(580)] = 0;
	JVS[MY_I(581)] = -B[MY_I_A(236)];
	JVS[MY_I(582)] = -B[MY_I_A(127)]-B[MY_I_A(186)]-B[MY_I_A(213)]-B[MY_I_A(237)];
	JVS[MY_I(583)] = 0;
	JVS[MY_I(584)] = -B[MY_I_A(128)];
	JVS[MY_I(585)] = 0;
	JVS[MY_I(586)] = B[MY_I_A(75)];
	JVS[MY_I(587)] = B[MY_I_A(15)];
	JVS[MY_I(588)] = 0;
	JVS[MY_I(589)] = 0;
	JVS[MY_I(590)] = 0;
	JVS[MY_I(591)] = 0;
	JVS[MY_I(592)] = 0;
	JVS[MY_I(593)] = -B[MY_I_A(183)];
	JVS[MY_I(594)] = -B[MY_I_A(210)];
	JVS[MY_I(595)] = B[MY_I_A(76)];
	JVS[MY_I(596)] = 0;
	JVS[MY_I(597)] = -B[MY_I_A(234)];
	JVS[MY_I(598)] = 0;
	JVS[MY_I(599)] = -B[MY_I_A(147)]-B[MY_I_A(184)]-B[MY_I_A(211)]-B[MY_I_A(235)];
	JVS[MY_I(600)] = -B[MY_I_A(148)];
	JVS[MY_I(601)] = 0;
	JVS[MY_I(602)] = B[MY_I_A(3)];
	JVS[MY_I(603)] = B[MY_I_A(23)];
	JVS[MY_I(604)] = -B[MY_I_A(143)];
	JVS[MY_I(605)] = -B[MY_I_A(145)];
	JVS[MY_I(606)] = -B[MY_I_A(135)];
	JVS[MY_I(607)] = -B[MY_I_A(151)];
	JVS[MY_I(608)] = -B[MY_I_A(264)];
	JVS[MY_I(609)] = -B[MY_I_A(133)];
	JVS[MY_I(610)] = -B[MY_I_A(129)];
	JVS[MY_I(611)] = -B[MY_I_A(131)];
	JVS[MY_I(612)] = -B[MY_I_A(141)];
	JVS[MY_I(613)] = -B[MY_I_A(29)];
	JVS[MY_I(614)] = -B[MY_I_A(137)];
	JVS[MY_I(615)] = -B[MY_I_A(149)];
	JVS[MY_I(616)] = -B[MY_I_A(35)];
	JVS[MY_I(617)] = -B[MY_I_A(125)];
	JVS[MY_I(618)] = -B[MY_I_A(45)];
	JVS[MY_I(619)] = B[MY_I_A(6)]-B[MY_I_A(51)]+B[MY_I_A(53)];
	JVS[MY_I(620)] = -B[MY_I_A(139)];
	JVS[MY_I(621)] = -B[MY_I_A(127)];
	JVS[MY_I(622)] = -B[MY_I_A(147)];
	JVS[MY_I(623)] = -B[MY_I_A(30)]-B[MY_I_A(36)]-B[MY_I_A(46)]-2*B[MY_I_A(47)]-B[MY_I_A(52)]-B[MY_I_A(126)]-B[MY_I_A(128)]-B[MY_I_A(130)]
			-B[MY_I_A(132)]-B[MY_I_A(134)]-B[MY_I_A(136)]-B[MY_I_A(138)]-B[MY_I_A(140)]-B[MY_I_A(142)]-B[MY_I_A(144)]-B[MY_I_A(146)]
			-B[MY_I_A(148)]-B[MY_I_A(150)]-B[MY_I_A(152)]-B[MY_I_A(265)];
	JVS[MY_I(624)] = B[MY_I_A(0)]+B[MY_I_A(24)]+B[MY_I_A(54)];
	JVS[MY_I(625)] = B[MY_I_A(124)];
	JVS[MY_I(626)] = B[MY_I_A(59)];
	JVS[MY_I(627)] = -B[MY_I_A(23)];
	JVS[MY_I(628)] = 0.65*B[MY_I_A(5)]+B[MY_I_A(39)]+B[MY_I_A(65)];
	JVS[MY_I(629)] = B[MY_I_A(121)];
	JVS[MY_I(630)] = B[MY_I_A(4)];
	JVS[MY_I(631)] = 0;
	JVS[MY_I(632)] = 0;
	JVS[MY_I(633)] = -B[MY_I_A(266)];
	JVS[MY_I(634)] = 0;
	JVS[MY_I(635)] = B[MY_I_A(143)];
	JVS[MY_I(636)] = B[MY_I_A(145)];
	JVS[MY_I(637)] = B[MY_I_A(135)];
	JVS[MY_I(638)] = 2*B[MY_I_A(151)]+B[MY_I_A(275)]+B[MY_I_A(277)]+2*B[MY_I_A(279)];
	JVS[MY_I(639)] = B[MY_I_A(264)];
	JVS[MY_I(640)] = B[MY_I_A(133)];
	JVS[MY_I(641)] = 0.92*B[MY_I_A(129)];
	JVS[MY_I(642)] = 0;
	JVS[MY_I(643)] = 0.76*B[MY_I_A(131)];
	JVS[MY_I(644)] = -B[MY_I_A(122)]+B[MY_I_A(141)];
	JVS[MY_I(645)] = B[MY_I_A(29)]-B[MY_I_A(49)];
	JVS[MY_I(646)] = B[MY_I_A(20)]+B[MY_I_A(115)];
	JVS[MY_I(647)] = 0;
	JVS[MY_I(648)] = B[MY_I_A(137)];
	JVS[MY_I(649)] = B[MY_I_A(149)];
	JVS[MY_I(650)] = B[MY_I_A(35)]-B[MY_I_A(37)];
	JVS[MY_I(651)] = B[MY_I_A(125)]+B[MY_I_A(276)];
	JVS[MY_I(652)] = -B[MY_I_A(61)]+B[MY_I_A(66)]+B[MY_I_A(116)];
	JVS[MY_I(653)] = B[MY_I_A(7)]+2*B[MY_I_A(51)]-B[MY_I_A(57)];
	JVS[MY_I(654)] = -B[MY_I_A(119)]+B[MY_I_A(139)]+B[MY_I_A(278)];
	JVS[MY_I(655)] = 0.964*B[MY_I_A(127)];
	JVS[MY_I(656)] = B[MY_I_A(147)];
	JVS[MY_I(657)] = B[MY_I_A(30)]+B[MY_I_A(36)]+2*B[MY_I_A(47)]+2*B[MY_I_A(52)]+B[MY_I_A(126)]+0.964*B[MY_I_A(128)]+0.92
			*B[MY_I_A(130)]+0.76*B[MY_I_A(132)]+B[MY_I_A(134)]+B[MY_I_A(136)]+B[MY_I_A(138)]+B[MY_I_A(140)]+B[MY_I_A(142)]
			+B[MY_I_A(144)]+B[MY_I_A(146)]+B[MY_I_A(148)]+B[MY_I_A(150)]+2*B[MY_I_A(152)]+B[MY_I_A(265)];
	JVS[MY_I(658)] = -B[MY_I_A(0)]-B[MY_I_A(24)]-B[MY_I_A(38)]-B[MY_I_A(50)]-B[MY_I_A(58)]-B[MY_I_A(62)]-B[MY_I_A(120)]-B[MY_I_A(123)]-B[MY_I_A(267)];
}

void Jac() {
	Njac++;
	dim3 gridDim, blockDim;
	int length;

	length = (39)*(19)*(39);

	blockDim.x = 128;
	gridDim.x = length / blockDim.x + (length % blockDim.x > 0);
	
	//timer_start(&metrics.odejac);
	cudaStartTimer(&timingjac, 0, 0);

	for(device = startDevice; device < endDevice; device++) {
		cudaSetDevice(device);

		dev_Jac_SP<<<gridDim, blockDim>>>(
			length, 
			DataDevice[device].d_VAR, DataDevice[device].d_FIX, 
			DataDevice[device].d_RCONST, DataDevice[device].Jac0);
	}

	for(device = startDevice; device < endDevice; device++) {
		cudaSetDevice(device);
	
		cudaDeviceSynchronize();
	}
	//timer_stop(&metrics.odejac);
	cudaStopTimer(&timingjac, 0, 0);
}

__global__ void dev_KppPrepare(int dom_dim, int * mask, int * mask2
#if 0
                   , double *vec_ghinv   // ZAP for bit for bit rel prev version, compute this on host
#endif
                   , double *H, int Direction, double gam, double * Ghimj)
{
        int i ;
        int p = blockIdx.x*blockDim.x + threadIdx.x;
        if(p >= dom_dim) return;
        if ( mask[p] == 0 || mask2[p] == 0 ) return ;

        for ( i = 0 ; i < NVAR ; i++ ) {
#if 0
          Ghimj[ MY_I(i) ] += vec_ghinv[p] ; //ZAP
#else
          Ghimj[ MY_I( i ) ] += ONE / (Direction * (H[p]) * gam);
#endif
        }
}

__host__ void KppPrepare(int *mask, int *mask2
#if 0
               , double * vec_ghinv //ZAP
#endif
               , double * H, int Direction, double gam, double * Ghimj) {

        dim3 gridDim, blockDim;

        int length = 39*39*19;
        blockDim.x = 128;
        gridDim.x = length / blockDim.x + (length % blockDim.x > 0);
	cudaSetDevice(0);
        dev_KppPrepare<<<gridDim, blockDim>>>(length,mask,mask2
#if 0
                                             ,vec_ghinv //ZAP
#endif
                                             ,H,Direction,gam,Ghimj);
}

static double * d_W ;

__global__ void dev_KppDecomp(int * vecMask, int * vecMask2, int length, double * JVS, int * retval) {

	extern __shared__ double share_W[];
	double * W;

	int j, k, jj, kk;
	int idx, lidx, uidx, lridx, uridx, didx;
	double a;

	// One-thread-per-cell decomposition
	int p = blockIdx.x*blockDim.x + threadIdx.x;
	retval[p] = 0;
	if(p >= length) return;

        if ( vecMask[p] == 0 || vecMask2[p] == 0 ) return ;
        int dom_dim = length ;

	W = &share_W[0];
	# define MY_I_S(i) ((i)*blockDim.x + threadIdx.x)

	for (k = 0; k < NVAR; k++) {
		// Get diagonal and lower/upper row indices
		didx  = LU_DIAG(k);

		lridx = LU_CROW(k);
		uridx = LU_CROW(k+1);

		if (JVS[MY_I(didx)] == 0.) {
			//return k + 1;
			retval[p] = k+1;
			return;
		}

		for (kk = lridx; kk < uridx; kk++) {
			idx = LU_ICOL(kk);
			W[MY_I_S(idx)] = JVS[MY_I(kk)];
		}
		for (kk = lridx; kk < didx; kk++) {
			j = LU_ICOL(kk);
			idx = LU_DIAG(j);
			a = -W[MY_I_S(j)] / JVS[MY_I(idx)];
			W[MY_I_S(j)] = -a;
			lidx = LU_DIAG(j);
			uidx = LU_CROW(j+1);
			for (jj = lidx + 1; jj < uidx; jj++) {
				idx = LU_ICOL(jj);
				W[MY_I_S(idx)] += a * JVS[MY_I(jj)];
			}
		}
		for (kk = lridx; kk < uridx; kk++) {
			idx = LU_ICOL(kk);
			JVS[MY_I(kk)] = W[MY_I_S(idx)];
		}
	}
}

__host__ int KppDecomp(
	int * d_vecMask, int * d_vecMask2, double * JVS) {

	static bool init = false;
#ifndef USE_CONSTANT_CACHE
	static cudaChannelFormatDesc desc_LU_DIAG = cudaCreateChannelDesc<int>();
	static cudaChannelFormatDesc desc_LU_CROW = cudaCreateChannelDesc<int>();
	static cudaChannelFormatDesc desc_LU_ICOL = cudaCreateChannelDesc<int>();
#endif
	static int * retval = 0;
	static int * d_retval = 0;
	static int length, size;

	int i, imax = -1;
	int max = std::numeric_limits<int>::min();
	dim3 gridDim, blockDim;

	//timer_start(&metrics.ludecomp);
	cudaStartTimer(&timingludecomp, 0, 0);

	if(!init) {
		length = 39*39*19;
		size   = length*sizeof(int);
		// Kernel return values
		retval = (int*)malloc(size);
		cudaMalloc((void**)&d_retval, size);
#ifdef USE_CONSTANT_CACHE
                copy_device_sparse_data() ;
#else
		// Bind textures
		cudaBindTexture(0, tex_LU_DIAG, dev_LU_DIAG, desc_LU_DIAG, CNVAR*sizeof(int));
		cudaBindTexture(0, tex_LU_CROW, dev_LU_CROW, desc_LU_CROW, CNVAR*sizeof(int));
		cudaBindTexture(0, tex_LU_ICOL, dev_LU_ICOL, desc_LU_ICOL, LU_NONZERO*sizeof(int));
#endif
		init = true;
	}

	// One-thread-per-cell decomposition
	// Shared memory is the limiting factor
        //blockDim.x = blockDim.x / 2 ;
	if(blockDim.x == 0) {
		fprintf(stderr, "Error: NVAR = %d is too big for shared memory.\n", NVAR);
		exit(1);
	}
	gridDim.x = length / blockDim.x + (length % blockDim.x > 0);

	// Launch kernel with dynamically-allocated shared memory
	cudaSetDevice(0);
	dev_KppDecomp<<<gridDim, blockDim, blockDim.x*NVAR*sizeof(double)>>>(d_vecMask, d_vecMask2, length, JVS, d_retval);

	// Get return values
	cudaMemcpy(retval, d_retval, size, cudaMemcpyDeviceToHost);

	// Reduce return values
	for(i=0; i<length; i++) {
		if ( retval[i] > max ) {
			max = retval[i] ;
			imax = i ; 
		} 
	}

	//timer_stop(&metrics.ludecomp);
	cudaStopTimer(&timingludecomp, 0, 0);

	return max;
}

void Decomp(double A[], int* ising) {
	*ising = KppDecomp(d_vecMask, d_vecMask2, A);
	Ndec++;
}

__global__ void dev_KppSolve(int * vecMask, int * vecMask2, int length, double * mm_JVS, double * mm_X) {
        extern __shared__ double shared_X[];
        double * X;

        int p = blockIdx.x*blockDim.x + threadIdx.x;

        if(p >= length) return;
// mask
        if(vecMask[p] == 0 || vecMask2[p] == 0 ) return ;

        X   = &shared_X[0];

        #pragma unroll
        for(int i=0; i<NVAR; i++)
                X[i*blockDim.x + threadIdx.x] = mm_X[i*length + p ];

	# define X(i)    X[(i)*blockDim.x + threadIdx.x]
	# define JVS(i)  mm_JVS[(i)*length + p]

	X(18) = X(18)-JVS(66)*X(6);
	X(21) = X(21)-JVS(79)*X(9)-JVS(80)*X(10);
	X(24) = X(24)-JVS(92)*X(14)-JVS(93)*X(21);
	X(25) = X(25)-JVS(104)*X(23);
	X(31) = X(31)-JVS(148)*X(21);
	X(32) = X(32)-JVS(156)*X(15)-JVS(157)*X(28);
	X(34) = X(34)-JVS(180)*X(9);
	X(35) = X(35)-JVS(187)*X(10);
	X(36) = X(36)-JVS(194)*X(26)-JVS(195)*X(27);
	X(37) = X(37)-JVS(204)*X(23)-JVS(205)*X(26)-JVS(206)*X(27)-JVS(207)
		 *X(28);
	X(38) = X(38)-JVS(216)*X(7)-JVS(217)*X(8)-JVS(218)*X(21)-JVS(219)
		 *X(22);
	X(39) = X(39)-JVS(229)*X(23);
	X(40) = X(40)-JVS(238)*X(7);
	X(41) = X(41)-JVS(245)*X(31)-JVS(246)*X(34)-JVS(247)*X(35)-JVS(248)
		 *X(36)-JVS(249)*X(38)-JVS(250)*X(39)-JVS(251)*X(40);
	X(42) = X(42)-JVS(267)*X(15)-JVS(268)*X(20)-JVS(269)*X(22)-JVS(270)
		 *X(23)-JVS(271)*X(26)-JVS(272)*X(27)-JVS(273)*X(28)-JVS(274)
		 *X(30)-JVS(275)*X(31)-JVS(276)*X(34)-JVS(277)*X(35)-JVS(278)
		 *X(36)-JVS(279)*X(37)-JVS(280)*X(38)-JVS(281)*X(39)-JVS(282)
		 *X(40);
	X(43) = X(43)-JVS(298)*X(8);
	X(44) = X(44)-JVS(305)*X(11)-JVS(306)*X(21)-JVS(307)*X(29)-JVS(308)
		 *X(34)-JVS(309)*X(35);
	X(45) = X(45)-JVS(318)*X(18)-JVS(319)*X(23)-JVS(320)*X(26)-JVS(321)
		 *X(27)-JVS(322)*X(28);
	X(46) = X(46)-JVS(329)*X(31)-JVS(330)*X(37)-JVS(331)*X(40)-JVS(332)
		 *X(43)-JVS(333)*X(45);
	X(47) = X(47)-JVS(343)*X(15)-JVS(344)*X(26)-JVS(345)*X(27)-JVS(346)
		 *X(28)-JVS(347)*X(36)-JVS(348)*X(37)-JVS(349)*X(39)-JVS(350)
		 *X(40)-JVS(351)*X(41)-JVS(352)*X(43)-JVS(353)*X(44)-JVS(354)
		 *X(45)-JVS(355)*X(46);
	X(48) = X(48)-JVS(368)*X(28)-JVS(369)*X(45);
	X(49) = X(49)-JVS(378)*X(32)-JVS(379)*X(40)-JVS(380)*X(43)-JVS(381)
		 *X(45)-JVS(382)*X(46)-JVS(383)*X(48);
	X(50) = X(50)-JVS(393)*X(4)-JVS(394)*X(9)-JVS(395)*X(10)-JVS(396)
		 *X(13)-JVS(397)*X(15)-JVS(398)*X(19)-JVS(399)*X(20)-JVS(400)
		 *X(21)-JVS(401)*X(23)-JVS(402)*X(25)-JVS(403)*X(26)-JVS(404)
		 *X(27)-JVS(405)*X(28)-JVS(406)*X(29)-JVS(407)*X(30)-JVS(408)
		 *X(31)-JVS(409)*X(33)-JVS(410)*X(34)-JVS(411)*X(35)-JVS(412)
		 *X(36)-JVS(413)*X(37)-JVS(414)*X(38)-JVS(415)*X(39)-JVS(416)
		 *X(40)-JVS(417)*X(41)-JVS(418)*X(42)-JVS(419)*X(43)-JVS(420)
		 *X(44)-JVS(421)*X(45)-JVS(422)*X(46)-JVS(423)*X(47)-JVS(424)
		 *X(48)-JVS(425)*X(49);
	X(51) = X(51)-JVS(435)*X(16)-JVS(436)*X(17)-JVS(437)*X(20)-JVS(438)
		 *X(26)-JVS(439)*X(27)-JVS(440)*X(28)-JVS(441)*X(31)-JVS(442)
		 *X(34)-JVS(443)*X(35)-JVS(444)*X(36)-JVS(445)*X(37)-JVS(446)
		 *X(38)-JVS(447)*X(39)-JVS(448)*X(40)-JVS(449)*X(43)-JVS(450)
		 *X(44)-JVS(451)*X(45)-JVS(452)*X(47)-JVS(453)*X(48)-JVS(454)
		 *X(49)-JVS(455)*X(50);
	X(52) = X(52)-JVS(464)*X(4)-JVS(465)*X(5)-JVS(466)*X(6)-JVS(467)*X(7)
		 -JVS(468)*X(8)-JVS(469)*X(9)-JVS(470)*X(10)-JVS(471)*X(12)
		 -JVS(472)*X(13)-JVS(473)*X(15)-JVS(474)*X(16)-JVS(475)*X(17)
		 -JVS(476)*X(19)-JVS(477)*X(20)-JVS(478)*X(21)-JVS(479)*X(22)
		 -JVS(480)*X(23)-JVS(481)*X(24)-JVS(482)*X(25)-JVS(483)*X(26)
		 -JVS(484)*X(27)-JVS(485)*X(28)-JVS(486)*X(29)-JVS(487)*X(30)
		 -JVS(488)*X(32)-JVS(489)*X(33)-JVS(490)*X(34)-JVS(491)*X(35)
		 -JVS(492)*X(40)-JVS(493)*X(41)-JVS(494)*X(42)-JVS(495)*X(43)
		 -JVS(496)*X(44)-JVS(497)*X(45)-JVS(498)*X(46)-JVS(499)*X(47)
		 -JVS(500)*X(48)-JVS(501)*X(49)-JVS(502)*X(50)-JVS(503)*X(51);
	X(53) = X(53)-JVS(511)*X(14)-JVS(512)*X(19)-JVS(513)*X(21)-JVS(514)
		 *X(22)-JVS(515)*X(23)-JVS(516)*X(24)-JVS(517)*X(26)-JVS(518)
		 *X(27)-JVS(519)*X(28)-JVS(520)*X(29)-JVS(521)*X(30)-JVS(522)
		 *X(33)-JVS(523)*X(34)-JVS(524)*X(35)-JVS(525)*X(42)-JVS(526)
		 *X(43)-JVS(527)*X(44)-JVS(528)*X(45)-JVS(529)*X(47)-JVS(530)
		 *X(48)-JVS(531)*X(49)-JVS(532)*X(50)-JVS(533)*X(51)-JVS(534)
		 *X(52);
	X(54) = X(54)-JVS(541)*X(17)-JVS(542)*X(22)-JVS(543)*X(31)-JVS(544)
		 *X(32)-JVS(545)*X(33)-JVS(546)*X(34)-JVS(547)*X(35)-JVS(548)
		 *X(36)-JVS(549)*X(37)-JVS(550)*X(38)-JVS(551)*X(39)-JVS(552)
		 *X(40)-JVS(553)*X(43)-JVS(554)*X(44)-JVS(555)*X(45)-JVS(556)
		 *X(46)-JVS(557)*X(47)-JVS(558)*X(48)-JVS(559)*X(49)-JVS(560)
		 *X(50)-JVS(561)*X(51)-JVS(562)*X(52)-JVS(563)*X(53);
	X(55) = X(55)-JVS(569)*X(15)-JVS(570)*X(41)-JVS(571)*X(43)-JVS(572)
		 *X(44)-JVS(573)*X(45)-JVS(574)*X(46)-JVS(575)*X(48)-JVS(576)
		 *X(49)-JVS(577)*X(50)-JVS(578)*X(51)-JVS(579)*X(52)-JVS(580)
		 *X(53)-JVS(581)*X(54);
	X(56) = X(56)-JVS(586)*X(5)-JVS(587)*X(32)-JVS(588)*X(40)-JVS(589)
		 *X(43)-JVS(590)*X(45)-JVS(591)*X(46)-JVS(592)*X(48)-JVS(593)
		 *X(50)-JVS(594)*X(51)-JVS(595)*X(52)-JVS(596)*X(53)-JVS(597)
		 *X(54)-JVS(598)*X(55);
	X(57) = X(57)-JVS(602)*X(12)-JVS(603)*X(18)-JVS(604)*X(34)-JVS(605)
		 *X(35)-JVS(606)*X(36)-JVS(607)*X(37)-JVS(608)*X(38)-JVS(609)
		 *X(39)-JVS(610)*X(40)-JVS(611)*X(43)-JVS(612)*X(44)-JVS(613)
		 *X(45)-JVS(614)*X(48)-JVS(615)*X(49)-JVS(616)*X(50)-JVS(617)
		 *X(51)-JVS(618)*X(52)-JVS(619)*X(53)-JVS(620)*X(54)-JVS(621)
		 *X(55)-JVS(622)*X(56);
	X(58) = X(58)-JVS(625)*X(11)-JVS(626)*X(14)-JVS(627)*X(18)-JVS(628)
		 *X(19)-JVS(629)*X(22)-JVS(630)*X(24)-JVS(631)*X(29)-JVS(632)
		 *X(30)-JVS(633)*X(31)-JVS(634)*X(33)-JVS(635)*X(34)-JVS(636)
		 *X(35)-JVS(637)*X(36)-JVS(638)*X(37)-JVS(639)*X(38)-JVS(640)
		 *X(39)-JVS(641)*X(40)-JVS(642)*X(42)-JVS(643)*X(43)-JVS(644)
		 *X(44)-JVS(645)*X(45)-JVS(646)*X(46)-JVS(647)*X(47)-JVS(648)
		 *X(48)-JVS(649)*X(49)-JVS(650)*X(50)-JVS(651)*X(51)-JVS(652)
		 *X(52)-JVS(653)*X(53)-JVS(654)*X(54)-JVS(655)*X(55)-JVS(656)
		 *X(56)-JVS(657)*X(57);
	X(58) = X(58)/JVS(658);
	X(57) = (X(57)-JVS(624)*X(58))/(JVS(623));
	X(56) = (X(56)-JVS(600)*X(57)-JVS(601)*X(58))/(JVS(599));
	X(55) = (X(55)-JVS(583)*X(56)-JVS(584)*X(57)-JVS(585)*X(58))
		 /(JVS(582));
	X(54) = (X(54)-JVS(565)*X(55)-JVS(566)*X(56)-JVS(567)*X(57)-JVS(568)
		 *X(58))/(JVS(564));
	X(53) = (X(53)-JVS(536)*X(54)-JVS(537)*X(55)-JVS(538)*X(56)-JVS(539)
		 *X(57)-JVS(540)*X(58))/(JVS(535));
	X(52) = (X(52)-JVS(505)*X(53)-JVS(506)*X(54)-JVS(507)*X(55)-JVS(508)
		 *X(56)-JVS(509)*X(57)-JVS(510)*X(58))/(JVS(504));
	X(51) = (X(51)-JVS(457)*X(52)-JVS(458)*X(53)-JVS(459)*X(54)-JVS(460)
		 *X(55)-JVS(461)*X(56)-JVS(462)*X(57)-JVS(463)*X(58))
		 /(JVS(456));
	X(50) = (X(50)-JVS(427)*X(51)-JVS(428)*X(52)-JVS(429)*X(53)-JVS(430)
		 *X(54)-JVS(431)*X(55)-JVS(432)*X(56)-JVS(433)*X(57)-JVS(434)
		 *X(58))/(JVS(426));
	X(49) = (X(49)-JVS(385)*X(50)-JVS(386)*X(51)-JVS(387)*X(52)-JVS(388)
		 *X(53)-JVS(389)*X(54)-JVS(390)*X(55)-JVS(391)*X(57)-JVS(392)
		 *X(58))/(JVS(384));
	X(48) = (X(48)-JVS(371)*X(50)-JVS(372)*X(51)-JVS(373)*X(52)-JVS(374)
		 *X(53)-JVS(375)*X(54)-JVS(376)*X(57)-JVS(377)*X(58))
		 /(JVS(370));
	X(47) = (X(47)-JVS(357)*X(48)-JVS(358)*X(49)-JVS(359)*X(50)-JVS(360)
		 *X(51)-JVS(361)*X(52)-JVS(362)*X(53)-JVS(363)*X(54)-JVS(364)
		 *X(55)-JVS(365)*X(56)-JVS(366)*X(57)-JVS(367)*X(58))
		 /(JVS(356));
	X(46) = (X(46)-JVS(335)*X(50)-JVS(336)*X(51)-JVS(337)*X(52)-JVS(338)
		 *X(53)-JVS(339)*X(54)-JVS(340)*X(55)-JVS(341)*X(57)-JVS(342)
		 *X(58))/(JVS(334));
	X(45) = (X(45)-JVS(324)*X(50)-JVS(325)*X(52)-JVS(326)*X(53)-JVS(327)
		 *X(57)-JVS(328)*X(58))/(JVS(323));
	X(44) = (X(44)-JVS(311)*X(50)-JVS(312)*X(51)-JVS(313)*X(52)-JVS(314)
		 *X(53)-JVS(315)*X(54)-JVS(316)*X(57)-JVS(317)*X(58))
		 /(JVS(310));
	X(43) = (X(43)-JVS(300)*X(50)-JVS(301)*X(51)-JVS(302)*X(52)-JVS(303)
		 *X(54)-JVS(304)*X(57))/(JVS(299));
	X(42) = (X(42)-JVS(284)*X(43)-JVS(285)*X(44)-JVS(286)*X(45)-JVS(287)
		 *X(48)-JVS(288)*X(49)-JVS(289)*X(50)-JVS(290)*X(51)-JVS(291)
		 *X(52)-JVS(292)*X(53)-JVS(293)*X(54)-JVS(294)*X(55)-JVS(295)
		 *X(56)-JVS(296)*X(57)-JVS(297)*X(58))/(JVS(283));
	X(41) = (X(41)-JVS(253)*X(43)-JVS(254)*X(44)-JVS(255)*X(45)-JVS(256)
		 *X(48)-JVS(257)*X(49)-JVS(258)*X(50)-JVS(259)*X(51)-JVS(260)
		 *X(52)-JVS(261)*X(53)-JVS(262)*X(54)-JVS(263)*X(55)-JVS(264)
		 *X(56)-JVS(265)*X(57)-JVS(266)*X(58))/(JVS(252));
	X(40) = (X(40)-JVS(240)*X(50)-JVS(241)*X(51)-JVS(242)*X(52)-JVS(243)
		 *X(54)-JVS(244)*X(57))/(JVS(239));
	X(39) = (X(39)-JVS(231)*X(45)-JVS(232)*X(50)-JVS(233)*X(51)-JVS(234)
		 *X(52)-JVS(235)*X(53)-JVS(236)*X(54)-JVS(237)*X(57))
		 /(JVS(230));
	X(38) = (X(38)-JVS(221)*X(44)-JVS(222)*X(50)-JVS(223)*X(51)-JVS(224)
		 *X(52)-JVS(225)*X(53)-JVS(226)*X(54)-JVS(227)*X(57)-JVS(228)
		 *X(58))/(JVS(220));
	X(37) = (X(37)-JVS(209)*X(45)-JVS(210)*X(50)-JVS(211)*X(51)-JVS(212)
		 *X(52)-JVS(213)*X(53)-JVS(214)*X(54)-JVS(215)*X(57))
		 /(JVS(208));
	X(36) = (X(36)-JVS(197)*X(45)-JVS(198)*X(50)-JVS(199)*X(51)-JVS(200)
		 *X(52)-JVS(201)*X(53)-JVS(202)*X(54)-JVS(203)*X(57))
		 /(JVS(196));
	X(35) = (X(35)-JVS(189)*X(50)-JVS(190)*X(51)-JVS(191)*X(52)-JVS(192)
		 *X(54)-JVS(193)*X(57))/(JVS(188));
	X(34) = (X(34)-JVS(182)*X(50)-JVS(183)*X(51)-JVS(184)*X(52)-JVS(185)
		 *X(54)-JVS(186)*X(57))/(JVS(181));
	X(33) = (X(33)-JVS(171)*X(34)-JVS(172)*X(35)-JVS(173)*X(44)-JVS(174)
		 *X(49)-JVS(175)*X(51)-JVS(176)*X(52)-JVS(177)*X(53)-JVS(178)
		 *X(54)-JVS(179)*X(57))/(JVS(170));
	X(32) = (X(32)-JVS(159)*X(40)-JVS(160)*X(43)-JVS(161)*X(45)-JVS(162)
		 *X(46)-JVS(163)*X(48)-JVS(164)*X(51)-JVS(165)*X(52)-JVS(166)
		 *X(53)-JVS(167)*X(54)-JVS(168)*X(55)-JVS(169)*X(57))
		 /(JVS(158));
	X(31) = (X(31)-JVS(150)*X(50)-JVS(151)*X(51)-JVS(152)*X(52)-JVS(153)
		 *X(53)-JVS(154)*X(54)-JVS(155)*X(58))/(JVS(149));
	X(30) = (X(30)-JVS(141)*X(34)-JVS(142)*X(44)-JVS(143)*X(51)-JVS(144)
		 *X(52)-JVS(145)*X(53)-JVS(146)*X(54)-JVS(147)*X(57))
		 /(JVS(140));
	X(29) = (X(29)-JVS(133)*X(34)-JVS(134)*X(35)-JVS(135)*X(51)-JVS(136)
		 *X(52)-JVS(137)*X(53)-JVS(138)*X(54)-JVS(139)*X(57))
		 /(JVS(132));
	X(28) = (X(28)-JVS(129)*X(45)-JVS(130)*X(52)-JVS(131)*X(53))
		 /(JVS(128));
	X(27) = (X(27)-JVS(125)*X(45)-JVS(126)*X(52)-JVS(127)*X(53))
		 /(JVS(124));
	X(26) = (X(26)-JVS(121)*X(45)-JVS(122)*X(52)-JVS(123)*X(53))
		 /(JVS(120));
	X(25) = (X(25)-JVS(106)*X(26)-JVS(107)*X(27)-JVS(108)*X(28)-JVS(109)
		 *X(30)-JVS(110)*X(33)-JVS(111)*X(42)-JVS(112)*X(44)-JVS(113)
		 *X(45)-JVS(114)*X(47)-JVS(115)*X(51)-JVS(116)*X(52)-JVS(117)
		 *X(53)-JVS(118)*X(54)-JVS(119)*X(57))/(JVS(105));
	X(24) = (X(24)-JVS(95)*X(29)-JVS(96)*X(30)-JVS(97)*X(33)-JVS(98)
		 *X(42)-JVS(99)*X(47)-JVS(100)*X(50)-JVS(101)*X(52)-JVS(102)
		 *X(53)-JVS(103)*X(58))/(JVS(94));
	X(23) = (X(23)-JVS(89)*X(45)-JVS(90)*X(52)-JVS(91)*X(53))/(JVS(88));
	X(22) = (X(22)-JVS(85)*X(52)-JVS(86)*X(54)-JVS(87)*X(58))/(JVS(84));
	X(21) = (X(21)-JVS(82)*X(52)-JVS(83)*X(53))/(JVS(81));
	X(20) = (X(20)-JVS(76)*X(50)-JVS(77)*X(51)-JVS(78)*X(52))/(JVS(75));
	X(19) = (X(19)-JVS(72)*X(50)-JVS(73)*X(52)-JVS(74)*X(58))/(JVS(71));
	X(18) = (X(18)-JVS(68)*X(45)-JVS(69)*X(53)-JVS(70)*X(58))/(JVS(67));
	X(17) = (X(17)-JVS(63)*X(50)-JVS(64)*X(52)-JVS(65)*X(54))/(JVS(62));
	X(16) = (X(16)-JVS(58)*X(27)-JVS(59)*X(28)-JVS(60)*X(45)-JVS(61)
		 *X(52))/(JVS(57));
	X(15) = (X(15)-JVS(56)*X(52))/(JVS(55));
	X(14) = (X(14)-JVS(53)*X(53)-JVS(54)*X(58))/(JVS(52));
	X(13) = (X(13)-JVS(50)*X(50)-JVS(51)*X(52))/(JVS(49));
	X(12) = (X(12)-JVS(47)*X(52)-JVS(48)*X(57))/(JVS(46));
	X(11) = (X(11)-JVS(44)*X(44)-JVS(45)*X(58))/(JVS(43));
	X(10) = (X(10)-JVS(42)*X(52))/(JVS(41));
	X(9) = (X(9)-JVS(40)*X(52))/(JVS(39));
	X(8) = (X(8)-JVS(38)*X(52))/(JVS(37));
	X(7) = (X(7)-JVS(36)*X(52))/(JVS(35));
	X(6) = (X(6)-JVS(34)*X(45))/(JVS(33));
	X(5) = (X(5)-JVS(32)*X(52))/(JVS(31));
	X(4) = (X(4)-JVS(30)*X(52))/(JVS(29));
	X(3) = (X(3)-JVS(27)*X(25)-JVS(28)*X(52))/(JVS(26));
	X(2) = (X(2)-JVS(10)*X(26)-JVS(11)*X(27)-JVS(12)*X(28)-JVS(13)*X(36)
		-JVS(14)*X(37)-JVS(15)*X(39)-JVS(16)*X(40)-JVS(17)*X(43)
		-JVS(18)*X(44)-JVS(19)*X(45)-JVS(20)*X(48)-JVS(21)*X(49)
		-JVS(22)*X(51)-JVS(23)*X(54)-JVS(24)*X(55)-JVS(25)*X(56))
		/(JVS(9));
	X(1) = (X(1)-JVS(4)*X(23)-JVS(5)*X(26)-JVS(6)*X(27)-JVS(7)*X(28)
		-JVS(8)*X(45))/(JVS(3));
	X(0) = (X(0)-JVS(1)*X(4)-JVS(2)*X(52))/(JVS(0));

#undef X
	#pragma unroll
	for(int i=0; i<NVAR; i++)
                mm_X[i*length + p ] = X[i*blockDim.x + threadIdx.x] ;
}

__host__ void KppSolve(int * d_vecMask, int * d_vecMask2, double * d_JVS, double * d_X) {

	dim3 gridDim, blockDim;
	int length;

	length = 39*39*19;

	// One-thread-per-cell decomposition
	// Shared memory is the limiting factor

        //blockDim.x = blockDim.x / 2 ;
	if(blockDim.x == 0) {
		fprintf(stderr, "Error: NVAR = %d is too big for shared memory.\n", NVAR);
		exit(1);
	}
	gridDim.x = length / blockDim.x + (length % blockDim.x > 0);

	//timer_start(&metrics.lusolve);
	// Launch kernel with dynamically-allocated shared memory
	cudaSetDevice(0);
	dev_KppSolve<<<gridDim, blockDim, blockDim.x*NVAR*sizeof(double)>>>(d_vecMask, d_vecMask2, length, d_JVS, d_X);

	//timer_stop(&metrics.lusolve);
}

void Solve(double A[], double b[]) {
	KppSolve(d_vecMask, d_vecMask2, A, b);
	Nsol++;
}

char ros_PrepareMatrix(
	double* d_H, 
	int Direction,  double gam, double * Jac0,
	double * Ghimj, int * Pivot) {
	
	int ising, Nconsecutive;
	int length = LU_NONZERO * 39 * 19 * 39;

	Nconsecutive = 0;

	while(1) {
		WCOPY(length, Jac0, Ghimj);
		WSCAL(length, (-ONE), Ghimj);

		KppPrepare(d_vecMask, d_vecMask2, d_H, Direction, gam, Ghimj);

		// Compute LU decomposition
		Decomp(Ghimj, &ising);

		if (ising == 0) {
			// if successful done
			return 0; // Singular = false
		} else { // ising .ne. 0
#if 1
			return 1; // Singular = true
#else
			// if unsuccessful half the step size; if 5 consecutive fails return
			Nsng++;
			Nconsecutive++;
			fprintf(stderr,"\nWarning: LU Decomposition returned ising = %d\n", ising);
			if (Nconsecutive <= 5) { // Less than 5 consecutive failed LUs
				*H = (*H) * HALF;
			} else { // More than 5 consecutive failed LUs
				return 1; // Singular = true
			} // end if  Nconsecutive
#endif
		}
	}
}

__global__ void dev_acceptStep(
	int length, 
	double * Y, double * Ynew, double * Yerr, 
	double **K_in,
	double * H, double * T,
	double * AbsTol, double * RelTol,
	int * vecMask, int * vecMask2,
	int * RejectLastH, int * RejectMoreH,
	int ros_S,
	double Hmin, double ros_ELO,
	double FacSafe, double FacMax, double FacMin, double FacRej, int Direction
#ifdef RETURN_ERROR
	,double * vecErr
#endif
#ifdef RETURN_DEBUG_STAT 
	,double * stat
#endif
) {

	int ii, jj ;
	int i = threadIdx.x;
	int j = blockIdx.y;
	int k = blockIdx.x;
	int p = j*gridDim.x*blockDim.x + k*blockDim.x + i;
	int dom_dim = length ;
   // local
	double Err, Fac, Scale, Ymax, Hnew ;
	if ( p >= length ) return ;

	if ( vecMask[p] != 0 && vecMask2[p] != 0 ) {
#if 0
	for ( ii = 0 ; ii < NVAR ; ii++ ) {
		Ynew[MY_I(ii)] = Y[MY_I(ii)] ;
	}
	for ( jj = 0 ; jj < ros_S ; jj++ ) {
		for ( ii = 0 ; ii < NVAR ; ii++ ) {
			Ynew[MY_I(ii)] += const_ros_M[jj] * K_in[jj][MY_I(ii)] ;
		}
	}

	for ( ii = 0 ; ii < NVAR ; ii++ ) {
		Yerr[MY_I(ii)] = 0. ;
	}
	for ( jj = 0 ; jj < ros_S ; jj++ ) {
		for ( ii = 0 ; ii < NVAR ; ii++ ) {
			Yerr[MY_I(ii)] += const_ros_E[jj] * K_in[jj][MY_I(ii)] ;
		}
	}
	Err = 0.0 ;
	for ( ii = 0 ; ii < NVAR ; ii++ ) {
		Ymax  = MAX(ABS(Y[MY_I(ii)]), ABS(Ynew[MY_I(ii)]));
		Scale = AbsTol[ii] + RelTol[ii] * Ymax;
		Err   += (Yerr[MY_I(ii)] * Yerr[MY_I(ii)]) / (Scale * Scale);
	}
#else
	double K0,K1,K2 ;
	Err = 0.0 ;
	#pragma unroll
	for ( ii = 0 ; ii < NVAR ; ii++ ) {
		K0 = K_in[0][MY_I(ii)] ;
		K1 = K_in[1][MY_I(ii)] ;
		K2 = K_in[2][MY_I(ii)] ;
		Ynew[MY_I(ii)] = const_ros_M[0] * K0 + const_ros_M[1] * K1 + const_ros_M[2] * K2 + Y[MY_I(ii)] ;
		Yerr[MY_I(ii)] = const_ros_E[0] * K0 + const_ros_E[1] * K1 + const_ros_E[2] * K2 ;
		Ymax  = MAX(ABS(Y[MY_I(ii)]), ABS(Ynew[MY_I(ii)]));
		//Scale = AbsTol[ii] + RelTol[ii] * Ymax;
		Scale = 1.0 + 1.0e-3 * Ymax;
		Err   += (Yerr[MY_I(ii)] * Yerr[MY_I(ii)]) / (Scale * Scale);
	}
#endif
	Err = sqrt(Err/(double)NVAR);
	Fac = MIN(FacMax,MAX(FacMin,FacSafe/pow(Err,ONE/ros_ELO)));
	Hnew = H[p]*Fac ; 
#ifdef RETURN_ERROR
	vecErr[p] = Err ;
#endif
	if ( (Err <= ONE) || (H[p] <= Hmin)) {
		for ( ii = 0 ; ii < NVAR ; ii++ ) {
			Y[MY_I(ii)] = Ynew[MY_I(ii)] ;
		}
		T[p] += Direction * H[p] ;
		if ( RejectLastH[p] ) Hnew = MIN(Hnew,H[p]) ;
		RejectLastH[p] = 0 ;
		RejectMoreH[p] = 0 ;
		H[p] = Hnew ;
		vecMask2[p] = 0 ;
	} else {
            if ( RejectMoreH[p] ) Hnew = H[p] * FacRej ;
            RejectMoreH[p] = RejectLastH[p] ;
            RejectLastH[p] = 1 ;
            H[p] = Hnew ;
            vecMask2[p] = 1 ;
	}
	}
}

__host__ int acceptStep(
	double * Y, double * Ynew, double * Yerr,
	double **K, double * vecH, double * vecT,
	double * ros_M, double * ros_E,
	double * AbsTol, double * RelTol,
	int * d_vecMask, int * d_vecMask2,
	int * RejectLastH, int * RejectMoreH,
	int ros_S ,double Hmin, double ros_ELO,
	double FacSafe, double FacMax, 
	double FacMin, double FacRej, int Direction) {

	int retval;
	int length;
	int i;
#ifdef RETURN_DEBUG_STAT
        double stat[1000] ;
#endif
	dim3 gridDim, blockDim;
        int * vecMask, * vecMask2 ;
        // One-thread-per-cell decomposition
#if 0
        gridDim.y  = 39;
        gridDim.x  = 19;
        blockDim.x = 39;
        length  = gridDim.y*gridDim.x*blockDim.x;
#else
	length  = 39*39*19 ;
# ifdef  DOUBLE_PRECISION
	blockDim.x = 128 ;
# else
	blockDim.x = 512 ;
# endif
	gridDim.x  = length/blockDim.x*blockDim.x + 1;
	gridDim.y  = 1 ;
#endif
	if ( first_accept_step ) {
		float tmp[4]; 
		tmp[0] = ros_M[0] ; tmp[1] = ros_M[1] ;tmp[2] = ros_M[2] ;tmp[3] = ros_M[3] ;
		cudaMemcpyToSymbol(const_ros_M,tmp,(4)*sizeof(float),0,cudaMemcpyHostToDevice) ;
		tmp[0] = ros_E[0] ; tmp[1] = ros_E[1] ;tmp[2] = ros_E[2] ;tmp[3] = ros_E[3] ;
		cudaMemcpyToSymbol(const_ros_E,tmp,(4)*sizeof(float),0,cudaMemcpyHostToDevice) ;
#ifdef RETURN_DEBUG_STAT 
	cudaMalloc((void**)&d_stat_accept, 1000*sizeof(double));
#endif
          first_accept_step = 0 ;
        }

#ifdef RETURN_ERROR
        double *vecErr ;
        cudaMalloc((void**)&vecErr,length*sizeof(double));
#endif

        dev_acceptStep <<<gridDim, blockDim>>> ( 
                     length,  Y,  Ynew,  Yerr
                    ,K
                    ,vecH ,  vecT
                    ,AbsTol,  RelTol
                    ,d_vecMask,  d_vecMask2
                    ,RejectLastH,  RejectMoreH
                    ,ros_S
                    ,Hmin, ros_ELO
                    ,FacSafe, FacMax, FacMin, FacRej, Direction
#ifdef RETURN_ERROR
                    ,vecErr
#endif
#ifdef RETURN_DEBUG_STAT 
                    ,d_stat_accept    // DEBUGGING
#endif
                                              ) ;

	// Get return values
	vecMask = (int*)malloc(length*sizeof(int));
	vecMask2 = (int*)malloc(length*sizeof(int));
	cudaMemcpy(vecMask, d_vecMask, length*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(vecMask2, d_vecMask2, length*sizeof(int), cudaMemcpyDeviceToHost);
	retval = 0 ;
	int idx = -1  ;
	for ( i = 0 ; i < length ; i++ ) {
		if ( vecMask[i] != 0 && vecMask2[i] != 0 ) { retval = 1 ; idx = i ;break ; }
	}
	int cnt = 0 ;
	for ( i = 0 ; i < length ; i++ ) {
		if ( vecMask[i] != 0 && vecMask2[i] != 0 ) { cnt++ ; }
	}
          
	if ( retval != 0 ) {
		double * tmp = (double *)malloc(length*sizeof(double) ) ;
		cudaMemcpy(tmp, vecT, length*sizeof(int), cudaMemcpyDeviceToHost);
		double minT =  999999. ;
		double maxT = -999999. ;
		for ( i = 0 ; i < length ; i++ ) {
			if ( vecMask[i] != 0 && vecMask2[i] != 0 ) {
				minT = MIN(minT,tmp[i]) ;
				maxT = MAX(maxT,tmp[i]) ;
			}
		}
		cudaMemcpy(tmp, vecH, length*sizeof(int), cudaMemcpyDeviceToHost);
		double minH =  999999. ;
		double maxH = -999999. ;
		for ( i = 0 ; i < length ; i++ ) {
			if ( vecMask[i] != 0 && vecMask2[i] != 0 ) {
				minH = MIN(minH,tmp[i]) ;
				maxH = MAX(maxH,tmp[i]) ;
			}
		}
#ifdef RETURN_ERROR
		cudaMemcpy(tmp, vecErr, length*sizeof(int), cudaMemcpyDeviceToHost);
		double minErr =  999999. ;
		double maxErr = -999999. ;
		for ( i = 0 ; i < length ; i++ ) {
			if ( vecMask[i] != 0 && vecMask2[i] != 0 ) {
				minErr = MIN(minErr,tmp[i]) ;
			}
		}
		int idxMaxErr = -1 ;
		for ( i = 0 ; i < length ; i++ ) {
			if ( vecMask[i] != 0 && vecMask2[i] != 0 ) {
				if ( maxErr < tmp[i] ) { maxErr = tmp[i] ; idxMaxErr = i ;} 
			}
		}
		if ( retval != 0 ) {
			fprintf(stderr,"accept rejects step cnt %d idx %d minT %E maxT %E minH %E maxH %E minErr %E maxErr %E %d\n", cnt, idx, minT, maxT, minH, maxH, minErr, maxErr, idxMaxErr ) ;
		}
#  ifdef WRITE_ERROR_VECTOR
		fprintf(stderr,"ERROR_VECTOR: element mask1 mask2 error\n") ;
		for ( i = 0 ; i < length ; i++ ) {
			fprintf(stderr,"ERROR_VECTOR: %10d %5d %5d %25.12e\n",i,vecMask[i],vecMask2[i],tmp[i]) ;
		}
		fprintf(stderr,"Finished writing error vector. quitting.  To do a full run recompile without -DWRITE_ERROR_VECTOR in Makefile.\n"); 
		exit(9) ;
#  endif
		cudaFree(vecErr) ;
#else

#  ifndef SUPPRESS_STATS_OUTPUT
		if ( retval != 0 ) {
			fprintf(stderr,"accept rejects step cnt %d idx %d minT %E maxT %E minH %E maxH %E \n", cnt, idx, minT, maxT, minH, maxH ) ;
		}
#  endif
#endif
		free(tmp) ;
	}

	free(vecMask);
	free(vecMask2);
	return( retval == 0 );
}

int RosenbrockIntegrator(
	double Tstart, double Tend,
	double  * AbsTol, double * RelTol,
	int ros_S,
	double ros_M[], double ros_E[],
	double ros_A[], double ros_C[],
	double ros_Alpha[],double  ros_Gamma[],
	double ros_ELO, char ros_NewF[],
	// Input: integration parameters
	char Autonomous, char VectorTol,
	int Max_no_steps,
	double Roundoff, double Hmin, double Hmax, double Hstart,
	double FacMin, double FacMax, double FacRej, double FacSafe,
	// Output: time at which the solution is returned (T=Tend  if success) and last accepted step
	double *Texit, double *Hexit ) {

	double H, T, Hnew, HC, HG, Fac;
	double Err;

	int Direction, ioffset, i, j, istage;
	char RejectLastH, RejectMoreH;

	int dom_dim = 39 * 39 * 19;
	
	double * Yerr;
	double * Fcn0;
	double * dFdT;
	double * Jac0;
	double * Ghimj;
	double * K[ros_S];
	int    * Pivot;
	
	cudaSetDevice(0);
	// Allocate device memory
	cudaMalloc((void**)&Ynew,  NVAR * dom_dim * sizeof(double));
	cudaMalloc((void**)&Yerr,  NVAR * dom_dim * sizeof(double));
	cudaMalloc((void**)&Fcn0,  NVAR * dom_dim * sizeof(double));
	cudaMalloc((void**)&Fcn,   NVAR * dom_dim * sizeof(double));
	cudaMalloc((void**)&dFdT,  NVAR * dom_dim * sizeof(double));
	cudaMalloc((void**)&Jac0,  LU_NONZERO * dom_dim * sizeof(double));
	cudaMalloc((void**)&Ghimj, LU_NONZERO * dom_dim * sizeof(double));

        // added jm 20090802
	cudaMalloc((void**)&d_vecT,        dom_dim * sizeof(double));
	cudaMalloc((void**)&d_vecH,        dom_dim * sizeof(double));
	cudaMalloc((void**)&d_vecMask,     dom_dim * sizeof(int));
	cudaMalloc((void**)&d_vecMask2,    dom_dim * sizeof(int));
	cudaMalloc((void**)&d_RejectLastH, dom_dim * sizeof(int));
	cudaMalloc((void**)&d_RejectMoreH, dom_dim * sizeof(int));

	for (i = 0; i < ros_S; i++) {
		cudaMalloc((void**) &K[i], NVAR * dom_dim * sizeof(double));
	}

	double ** d_K ;
	cudaMalloc((void**) &d_K, ros_S *sizeof(double *));
        cudaMemcpy(d_K,K,ros_S*sizeof(double *),cudaMemcpyHostToDevice);

	Pivot = NULL;

	// INITIAL PREPARATIONS
	T = Tstart;
	*Hexit = 0.0;
	H = MIN(Hstart,Hmax);
	if (ABS(H) <= 10.0 * Roundoff)
		H = DeltaMin;

	if (Tend >= Tstart) {
		Direction = +1;
	} else {
		Direction = -1;
	}

	RejectLastH = 0;
	RejectMoreH = 0;

        cudaMemset(d_RejectLastH, 0, dom_dim * sizeof(int));
        cudaMemset(d_RejectMoreH, 0, dom_dim * sizeof(int));
        setVectorReal(T, d_vecT, dom_dim);
        setVectorReal(H, d_vecH, dom_dim);
	
	// Temp untuk pemecahan data Fcn0	
	double * temp1Fcn0 = (double *)malloc(852520);
	double * temp2Fcn0 = (double *)malloc(852520);
	
	// Temp untuk pemecahan data Jac0	
	double * temp1Jac0;
	double * temp2Jac0;
	cudaMallocHost((void**)&temp1Jac0, 9522220 * sizeof(double));
	cudaMallocHost((void**)&temp2Jac0, 9522220 * sizeof(double));
	
	// Time loop begins below
	while(moreToGo_timeloop(T, Tend, Roundoff, Direction, d_vecT, d_vecH, d_vecMask)) {
		cudaMemset(d_vecMask2, 1, dom_dim * sizeof(int));
	
		// Compute the function at current time
		Fun();
		
		cudaSetDevice(0);
		cudaMemcpy(temp1Fcn0, DataDevice[0].Fcn0, 852520, cudaMemcpyDeviceToHost);

		cudaSetDevice(1);
		cudaMemcpy(temp2Fcn0, DataDevice[1].Fcn0, 852520, cudaMemcpyDeviceToHost);

		for(device = startDevice; device < endDevice; device++) {
			cudaSetDevice(device);
	
			cudaDeviceSynchronize();
		}
	
		cudaSetDevice(0);
		int index = 0;
		for(i = 0; i < 852520; i++) {
			tempFcn0[index] = temp1Fcn0[i];
			index++;
		}
	
		for(i = 0; i < 852520; i++) {
			tempFcn0[index] = temp2Fcn0[i];
			index++;
		}

		for(i = 0; i < 1; i++) {
			tempFcn0[index] = 0;
			index++;
		}

		index = 0;

		cudaMemcpy(Fcn0, tempFcn0, 1705041, cudaMemcpyHostToDevice);
	
		// Compute the function derivative with respect to T
		if (!Autonomous) {
			ros_FunTimeDerivative(T, Roundoff, Fcn0, dFdT);
		}
		
		// Compute the Jacobian at current time
		Jac();
		
		cudaSetDevice(0);
		cudaMemcpyAsync(temp1Jac0, DataDevice[0].Jac0, 9522220, cudaMemcpyDeviceToHost, DataDevice[0].stream);


		cudaSetDevice(1);
		cudaMemcpyAsync(temp2Jac0, DataDevice[1].Jac0, 9522220, cudaMemcpyDeviceToHost, DataDevice[1].stream);


		for(device = startDevice; device < endDevice; device++) {
			cudaSetDevice(device);
	
			cudaDeviceSynchronize();
		}

		cudaSetDevice(0);
		for(i = 0; i < 9522220; i++) {
			tempJac0[i] = temp1Jac0[index];
			index++;
		}
		
		index = 0;
		for(i = 0; i < 9522220; i++) {
			tempJac0[i] = temp2Jac0[index];
			index++;
		}

		tempJac0[19044440] = 0;

		index = 0;
		
		cudaMemcpy(Jac0, tempJac0, 19044441, cudaMemcpyHostToDevice);
		
		// Repeat step calculation until current step accepted
		while(1){
			
			int rc;

			if((rc = ros_PrepareMatrix( d_vecH, Direction, ros_Gamma[0], Jac0, Ghimj, Pivot))){
				*Texit = T;
				return ros_ErrorMsg(-8, T, H);
			}
			
			// Asumsi step accepted
			// Compute the stages
			//for (istage = 1; istage <= ros_S; istage++) {
				//printf("loop %d\n", istage);
				// Current istage offset. Current istage vector is *K[ioffset]
				//ioffset = istage - 1;
				// For the 1st istage the function has been computed previously
				if (istage == 1) {
					WCOPY(NVAR*dom_dim, Fcn0, Fcn);
				} else { // istage>1 and a new function evaluation is needed at current istage
				//	if (ros_NewF[istage-1]) {
				//		WCOPY(NVAR*dom_dim, DataDevice[0].d_VAR, Ynew);
				//		for (j = 1; j <= istage-1; j++) {
				//			WAXPY(NVAR*dom_dim, ros_A[(istage-1)*(istage-2)/2+j-1], K[j-1], Ynew);
				//		}
				//		Tau = T + ros_Alpha[istage-1]*Direction*H;
				//		FunTau();
				//	} // end if ros_NewF(istage)
				} // end if istage

				//WCOPY(NVAR*dom_dim, Fcn, K[ioffset]);

				//for (j = 1; j <= istage-1; j++) {
					//WAXPY_HC(dom_dim, d_vecMask, d_vecMask2, ros_C[(istage-1)*(istage-2)/2+j-1], Direction, d_vecH, K[j-1], K[ioffset]);
				//} // for j

				//if ((!Autonomous) && (ros_Gamma[istage-1])) {
					//HG = Direction*H*ros_Gamma[istage-1];
					//WAXPY(NVAR*dom_dim, HG, dFdT, K[ioffset]);
				//} // end if !Autonomous

				//Solve(Ghimj, K[ioffset]);
			//}
			
			Nstp++;

			if(acceptStep(
					DataDevice[0].d_VAR, Ynew, Yerr, d_K,
					d_vecH, d_vecT,
					ros_M, ros_E,
					AbsTol, RelTol,
					d_vecMask, d_vecMask2,
					d_RejectLastH, d_RejectMoreH,
					ros_S ,Hmin, ros_ELO, FacSafe, FacMax, FacMin, FacRej, Direction)
			) {
				break;
			}
		}
	}

	// Clean up
	cudaFree(Ynew);
	cudaFree(Yerr);
	cudaFree(Fcn0);
	cudaFree(Fcn);
	cudaFree(dFdT);
	cudaFree(Jac0);
	cudaFree(Ghimj);

	for (i = 0; i < ros_S; i++) {
		cudaFree(K[i]);
	}

	cudaFree(d_vecT);
	cudaFree(d_vecH);
	cudaFree(d_vecMask);
	
	*Texit = T;	
	return 1;
}

int Rosenbrock(
	double Tstart, double Tend,
	double * AbsTol, double * RelTol,
	double RPAR[], int IPAR[]) {

	static const int Smax = 6;

	// The method parameters
	int Method, ros_S;
	double ros_M[Smax], ros_E[Smax];
	double ros_A[Smax * (Smax - 1) / 2], ros_C[Smax * (Smax - 1) / 2];
	double ros_Alpha[Smax], ros_Gamma[Smax], ros_ELO;
	char ros_NewF[Smax], ros_Name[12];
	// Local variables
	int Max_no_steps, IERR, i, UplimTol;
	char Autonomous, VectorTol;
	double Roundoff, FacMin, FacMax, FacRej, FacSafe;
	double Hmin, Hmax, Hstart, Hexit, Texit;

	// Device pointers
	double * d_AbsTol = 0;
	double * d_RelTol = 0;

	// Initialize statistics
	Nfun = IPAR[10];
	Njac = IPAR[11];
	Nstp = IPAR[12];
	Nacc = IPAR[13];
	Nrej = IPAR[14];
	Ndec = IPAR[15];
	Nsol = IPAR[16];
	Nsng = IPAR[17];

	// Autonomous or time dependent ODE. Default is time dependent.
	Autonomous = !(IPAR[0] == 0);

	// For Scalar tolerances (IPAR[1] != 0)  the code uses AbsTol[0] and RelTol[0]
	// For Vector tolerances (IPAR[1] == 0) the code uses AbsTol(1:NVAR) and RelTol(1:NVAR)
	if (IPAR[1] == 0) {
		VectorTol = 1;
		UplimTol = 59;
	} else {
		VectorTol = 0;
		UplimTol = 1;
	}

	// The maximum number of steps admitted
	if (IPAR[2] == 0)
		Max_no_steps = 100000;
	else
		Max_no_steps = IPAR[2];
	if (Max_no_steps < 0) {
		printf("\n User-selected max no. of steps: IPAR[2]=%d\n", IPAR[2]);
		return ros_ErrorMsg(-1, Tstart, ZERO);
	}

	//  The particular Rosenbrock method chosen
	if (IPAR[3] == 0)
		Method = 3;
	else
		Method = IPAR[3];
	if ((IPAR[3] < 1) || (IPAR[3] > 5)) {
		printf("\n User-selected Rosenbrock method: IPAR[3]=%d\n", IPAR[3]);
		return ros_ErrorMsg(-2, Tstart, ZERO);
	}

	// Unit Roundoff (1+Roundoff>1)
	Roundoff = WLAMCH('E');

	// Lower bound on the step size: (positive value)
	Hmin = RPAR[0];
	if (RPAR[0] < ZERO) {
		printf("\n User-selected Hmin: RPAR[0]=%e\n", RPAR[0]);
		return ros_ErrorMsg(-3, Tstart, ZERO);
	}
	// Upper bound on the step size: (positive value)
	if (RPAR[1] == ZERO)
		Hmax = ABS(Tend-Tstart);
	else
		Hmax = MIN(ABS(RPAR[1]),ABS(Tend-Tstart));
	if (RPAR[1] < ZERO) {
		printf("\n User-selected Hmax: RPAR[1]=%e\n", RPAR[1]);
		return ros_ErrorMsg(-3, Tstart, ZERO);
	}
	// Starting step size: (positive value)
	if (RPAR[2] == ZERO)
		Hstart = MAX(Hmin,DeltaMin);
	else
		Hstart = MIN(ABS(RPAR[2]),ABS(Tend-Tstart));
	if (RPAR[2] < ZERO) {
		printf("\n User-selected Hstart: RPAR[2]=%e\n", RPAR[2]);
		return ros_ErrorMsg(-3, Tstart, ZERO);
	}
	// Step size can be changed s.t.  FacMin < Hnew/Hexit < FacMax
	if (RPAR[3] == ZERO)
		FacMin = (double) 0.2;
	else
		FacMin = RPAR[3];
	if (RPAR[3] < ZERO) {
		printf("\n User-selected FacMin: RPAR[3]=%e\n", RPAR[3]);
		return ros_ErrorMsg(-4, Tstart, ZERO);
	}
	if (RPAR[4] == ZERO)
		FacMax = (double) 6.0;
	else
		FacMax = RPAR[4];
	if (RPAR[4] < ZERO) {
		printf("\n User-selected FacMax: RPAR[4]=%e\n", RPAR[4]);
		return ros_ErrorMsg(-4, Tstart, ZERO);
	}
	// FacRej: Factor to decrease step after 2 succesive rejections
	if (RPAR[5] == ZERO)
		FacRej = (double) 0.1;
	else
		FacRej = RPAR[5];
	if (RPAR[5] < ZERO) {
		printf("\n User-selected FacRej: RPAR[5]=%e\n", RPAR[5]);
		return ros_ErrorMsg(-4, Tstart, ZERO);
	}
	// FacSafe: Safety Factor in the computation of new step size
	if (RPAR[6] == ZERO)
		FacSafe = (double) 0.9;
	else
		FacSafe = RPAR[6];
	if (RPAR[6] < ZERO) {
		printf("\n User-selected FacSafe: RPAR[6]=%e\n", RPAR[6]);
		return ros_ErrorMsg(-4, Tstart, ZERO);
	}
	// Check if tolerances are reasonable
	for (i = 0; i < UplimTol; i++) {
		if ((AbsTol[i] <= ZERO) || (RelTol[i] <= 10.0 * Roundoff) || (RelTol[i]
				>= ONE)) {
			printf("\n  AbsTol[%d] = %e\n", i, AbsTol[i]);
			printf("\n  RelTol[%d] = %e\n", i, RelTol[i]);
			return ros_ErrorMsg(-5, Tstart, ZERO);
		}
	}

	// Initialize the particular Rosenbrock method
	switch (Method) {
	case 2:
		Ros3(&ros_S, ros_A, ros_C, ros_M, ros_E, ros_Alpha, ros_Gamma,
				ros_NewF, &ros_ELO, ros_Name);
		break;
	default:
		printf("\n Unknown Rosenbrock method: IPAR[3]= %d", Method);
		return ros_ErrorMsg(-2, Tstart, ZERO);
	}

	// Tolerances approved.  Upload to device.
	cudaMalloc((void**) &d_AbsTol, NVAR * sizeof(double));
	cudaMalloc((void**) &d_RelTol, NVAR * sizeof(double));
	cudaMemcpy(d_AbsTol, AbsTol, NVAR * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_RelTol, RelTol, NVAR * sizeof(double), cudaMemcpyHostToDevice);

	// Rosenbrock method
	
	IERR = RosenbrockIntegrator(
			Tstart, Tend,
			d_AbsTol, d_RelTol,
			ros_S, ros_M, ros_E, ros_A, ros_C,
			ros_Alpha, ros_Gamma, ros_ELO, ros_NewF,
			Autonomous, VectorTol, Max_no_steps, Roundoff,
			Hmin, Hmax, Hstart,
			FacMin, FacMax, FacRej, FacSafe,
			&Texit, &Hexit);
	
	cudaFree(d_AbsTol);
	cudaFree(d_RelTol);

	// Collect run statistics
	IPAR[10] = Nfun;
	IPAR[11] = Njac;
	IPAR[12] = Nstp;
	IPAR[13] = Nacc;
	IPAR[14] = Nrej;
	IPAR[15] = Ndec;
	IPAR[16] = Nsol;
	IPAR[17] = Nsng;
	// Last T and H
	RPAR[10] = Texit;
	RPAR[11] = Hexit;

	return IERR;
}

void INTEGRATE() {
	// Tolerance vectors
	// Initialized on host and uploaded to device
	double ATOL[NVAR];
	double RTOL[NVAR];

	for(i = 0; i < NVAR; i++) {
    		ATOL[i] = 1.0;
    		RTOL[i] = 1.0e-3;
    	}

	int IERR;

	int IPAR[20];
	double RPAR[20];
	int Ns=0, Na=0, Nr=0, Ng=0;
	
	for (i = 0; i < 20; i++) {
		IPAR[i] = 0;
		RPAR[i] = 0.0;
	}
	IPAR[0] = 1; // autonomous
	IPAR[3] = 2; // ros3
	RPAR[2] = 0.01*240; // starting step

	IERR = Rosenbrock(
			0.0, 240,
			ATOL, RTOL,
			RPAR, IPAR);

	Ns=IPAR[12];
	Na=IPAR[13];
	Nr=IPAR[14];
	Ng=IPAR[17];
	printf("\n Step=%d  Acc=%d  Rej=%d  Singular=%d\n", Ns, Na, Nr, Ng);

	if (IERR < 0) {
		printf("\n Rosenbrock: Unsucessful step at T=%g: IERR=%d\n", 0.0, IERR);
	}
}

int main() {
	openingWRFChemData();

	initializeWRFChemData();

	convertWRFChemData();

	recoverDeviceMemory();

	updatingCoefficientWRFChemData();

	INTEGRATE();

	printf("\nTiming ODE Fun = %f ms\n\n", timingfun.elapsed);
	
	printf("Timing ODE Jac = %f ms\n\n", timingjac.elapsed);

	printf("Timing LU Decomp = %f ms\n\n", timingludecomp.elapsed);

	return 0;
}
