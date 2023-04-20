/******************************************************************************
*
* Copyright (C) 2009 - 2014 Xilinx, Inc.  All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* Use of the Software is limited solely to applications:
* (a) running on a Xilinx device, or
* (b) that interact with a Xilinx device through a bus or interconnect.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
* XILINX  BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
* WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF
* OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*
* Except as contained in this notice, the name of the Xilinx shall not be used
* in advertising or otherwise to promote the sale, use or other dealings in
* this Software without prior written authorization from Xilinx.
*
******************************************************************************/

/*
 * helloworld.c: simple test application
 *
 * This application configures UART 16550 to baud rate 9600.
 * PS7 UART (Zynq) is not initialized by this application, since
 * bootrom/bsp configures it to baud rate 115200
 *
 * ------------------------------------------------
 * | UART TYPE   BAUD RATE                        |
 * ------------------------------------------------
 *   uartns550   9600
 *   uartlite    Configurable only in HW design
 *   ps7_uart    115200 (configured by bootrom/bsp)
 */


#include <stdio.h>
#include <unistd.h>
#include "platform.h"
#include "xil_printf.h"
// peripheral headers
#include "xparameters.h"
#include "xil_exception.h"
#include "xil_cache.h"
//#include "xintc.h"

#include "xgpiops.h"
#include "xttcps.h"
#include "xscugic.h"
// SPI controller headers

#define SHARED_RAM_POS_R0toR1 0x70000000
#define SHARED_RAM_POS_R1toR0 0x70100000
#define SHARED_RAM_POS_AtoR0 0x70200000
#define SHARED_RAM_POS_R0toA 0x70300000
#define SHARED_RAM_POS_AtoR1 0x70400000
#define SHARED_RAM_POS_R1toA 0x70500000


#define NS550CrossTest

#define TTC_TICK_DEVICE_ID	XPAR_XTTCPS_3_DEVICE_ID
#define TTC_TICK_INTR_ID	XPAR_XTTCPS_3_INTR

#define TTCPS_CLOCK_HZ		XPAR_XTTCPS_3_CLOCK_HZ

#define INTC_DEVICE_ID		XPAR_SCUGIC_SINGLE_DEVICE_ID

#define INTC		XScuGic
#define INTC_HANDLER	XScuGic_InterruptHandler

INTC IntcInstance;		/* Instance of the Interrupt Controller */
XScuGic InterruptController;  /* Interrupt controller instance */

static XTtcPs xTimerInstance;

#define TEST_BUFFER_SIZE 128
//#define TEST_BUFFER_SIZE 16

unsigned int real_Rx_Length=0;

#define QSIZE			1280		/* Length of queue data buffer */

typedef struct queue
{
    int q_inp;      /* Insert index */
    int q_outp;     /* Extract index */
    unsigned char q_buf[QSIZE];  /* Data buffer */
}QUEUE;

int queue_insert(QUEUE*, int);
int queue_init(QUEUE*);
int queue_extract(QUEUE*);
int queue_full(QUEUE*);
int queue_empty(QUEUE*);
int queue_status(QUEUE*);

static volatile int TotalSentCount;

int TTCTest(void);

// These prototype is for TTC testing............................

/* Set up routines for timer counters */
static int SetupTicker(void);
static int SetupTimer(int DeviceID);

static int SetupInterruptSystem(u16 IntcDeviceID, XScuGic *IntcInstancePtr);

static void TickHandler(void *CallBackRef, u32 StatusEvent);
static void PWMHandler(void *CallBackRef, u32 StatusEvent);


/*
 * Constants to set the basic operating parameters.
 * PWM_DELTA_DUTY is critical to the running time of the test. Smaller values
 * make the test run longer.
 */
#define	TICK_TIMER_FREQ_HZ	100  /* Tick timer counter's output frequency */
#define	PWM_OUT_FREQ		350  /* PWM timer counter's output frequency */

#define PWM_DELTA_DUTY	50 /* Initial and increment to duty cycle for PWM */
#define TICKS_PER_CHANGE_PERIOD TICK_TIMER_FREQ_HZ * 5 /* Tick signals PWM */

/**************************** Type Definitions *******************************/
typedef struct {
	u32 OutputHz;	/* Output frequency */
	XInterval Interval;	/* Interval value */
	u8 Prescaler;	/* Prescaler value */
	u16 Options;	/* Option settings */
} TmrCntrSetup;


/************************** Variable Definitions *****************************/
#define NUM_DEVICES 12U
static XTtcPs TtcPsInst[NUM_DEVICES];	/* Number of available timer counters */


//600Hz (300,600) 400Hz(200,400)...

static TmrCntrSetup SettingsTable[NUM_DEVICES] = {
	{300, 0, 0, 0},	/* Ticker timer counter initial setup, only output freq */
	{600, 0, 0, 0}, /* PWM timer counter initial setup, only output freq */
	{600, 0, 0, 0}, /* PWM timer counter initial setup, only output freq */
	{600, 0, 0, 0}, /* PWM timer counter initial setup, only output freq */
};


static u32 MatchValue;  /* Match value for PWM, set by PWM interrupt handler,
			updated by main test routine */

static volatile u32 PWM_UpdateFlag;	/* Flag used by Ticker to signal PWM */
static volatile u8 ErrorCount;		/* Errors seen at interrupt time */
static volatile u32 TickCount;		/* Ticker interrupts between PWM change */

// end of TTC prototype-----------------------------

unsigned char Test_Hzcnt=0;
unsigned char TM_txTest[1200];

void Inter_Timer_Intr(void);

int main()
{
	int * ToR0 = (int*) SHARED_RAM_POS_R1toR0;
	int * RtnR0 = (int*) SHARED_RAM_POS_R0toR1;
	int * RtnApu = (int*) SHARED_RAM_POS_AtoR1;
	int * ToApu = (int*) SHARED_RAM_POS_R1toA;

	ToR0[0] = 0;
	ToApu[0] = 0;


    Xil_DCacheFlush();

    //xil_printf("This is SIFC IO platform test \r\n");

    TTCTest();

    long counter = 0;
    while (1){
    	sleep(1);
    	//printf("alive\r\n");
    }

    return 0;
}



unsigned int counter = 0;
unsigned int g_counter = 0;
void Inter_Timer_Intr(void)
{

   static unsigned char D20Hz_cnt=0;
   static unsigned char D100Hz_cnt=0;
   static unsigned char D200Hz_cnt=0;
   int * ToR0 = (int*) SHARED_RAM_POS_R1toR0;
   int * RtnR0 = (int*) SHARED_RAM_POS_R0toR1;
   int * RtnApu = (int*) SHARED_RAM_POS_AtoR1;
   int * ToApu = (int*) SHARED_RAM_POS_R1toA;
   unsigned int i;


   for(i=0;i<128;i++)
   {
      ToR0[i] = ToR0[i] + 1 ;
      ToApu[i] = ToApu[i] + 3;
   }

   if (counter == 199){
	   //printf("Run RPU1 600 ticks reached. FromR0: %d FromAPU: %d \r\n", RtnR0[0], RtnApu[0]);
	   printf("1110930 Run RPU1 200 ticks reached(gcounter: %d ). to R0: %d Apu: %d from R0: %d Apu: %d\r\n",g_counter,ToR0[0], ToApu[0], RtnR0[0], RtnApu[0]);
	   counter = 0;
	   g_counter +=1;
   }else{
	   counter += 1;

   }

   Xil_DCacheFlush();
   if((D100Hz_cnt%6)==0)
   {

   }
   else if((D100Hz_cnt%6)==1)
   {

   }


   if((D200Hz_cnt%3)==0)
   {

   }
   else if((D200Hz_cnt%3)==1)
   {

   }


   if((D20Hz_cnt%20)==10)
   {

   }


   D20Hz_cnt++;
   D100Hz_cnt++;
   D200Hz_cnt++;

   if(D100Hz_cnt==6) D100Hz_cnt=0;
   if(D200Hz_cnt==3) D200Hz_cnt=0;
   if(D20Hz_cnt==20) D20Hz_cnt=0;

}


int TTCTest(){

	int Status;

	/*
	 * Make sure the interrupts are disabled, in case this is being run
	 * again after a failure.
	 */

	/*
	 * Connect the Intc to the interrupt subsystem such that interrupts can
	 * occur. This function is application specific.
	 */
	Status = SetupInterruptSystem(INTC_DEVICE_ID, &InterruptController);
	if (Status != XST_SUCCESS) {
		return XST_FAILURE;
	}
	//printf("setup interrupt IP\r\n");

	/*
	 * Set up the Ticker timer
	 */
	Status = SetupTicker();
	if (Status != XST_SUCCESS) {
		return Status;
	}
	//printf("setup ticker\r\n");

	return XST_SUCCESS;
}


/****************************************************************************/
/**
*
* This function sets up the Ticker timer.
*
* @param	None
*
* @return	XST_SUCCESS if everything sets up well, XST_FAILURE otherwise.
*
* @note		None
*
*****************************************************************************/
int SetupTicker(void)
{
	int Status;
	TmrCntrSetup *TimerSetup;
	XTtcPs *TtcPsTick;

	TimerSetup = &(SettingsTable[TTC_TICK_DEVICE_ID]);

	/*
	 * Set up appropriate options for Ticker: interval mode without
	 * waveform output.
	 */
	TimerSetup->Options |= (XTTCPS_OPTION_INTERVAL_MODE |
					      XTTCPS_OPTION_WAVE_DISABLE);

	/*
	 * Calling the timer setup routine
	 *  . initialize device
	 *  . set options
	 */
	Status = SetupTimer(TTC_TICK_DEVICE_ID);
	if(Status != XST_SUCCESS) {
		return Status;
	}

	TtcPsTick = &(TtcPsInst[TTC_TICK_DEVICE_ID]);

	/*
	 * Connect to the interrupt controller
	 */
	Status = XScuGic_Connect(&InterruptController, TTC_TICK_INTR_ID,
		(Xil_ExceptionHandler)XTtcPs_InterruptHandler, (void *)TtcPsTick);
	if (Status != XST_SUCCESS) {
		return XST_FAILURE;
	}

	XTtcPs_SetStatusHandler(&(TtcPsInst[TTC_TICK_DEVICE_ID]), &(TtcPsInst[TTC_TICK_DEVICE_ID]),
		              (XTtcPs_StatusHandler)TickHandler);

	/*
	 * Enable the interrupt for the Timer counter
	 */
	XScuGic_Enable(&InterruptController, TTC_TICK_INTR_ID);

	/*
	 * Enable the interrupts for the tick timer/counter
	 * We only care about the interval timeout.
	 */
	XTtcPs_EnableInterrupts(TtcPsTick, XTTCPS_IXR_INTERVAL_MASK);

	/*
	 * Start the tick timer/counter
	 */
	XTtcPs_Start(TtcPsTick);

	return Status;
}


/****************************************************************************/
/**
*
* This function sets up a timer counter device, using the information in its
* setup structure.
*  . initialize device
*  . set options
*  . set interval and prescaler value for given output frequency.
*
* @param	DeviceID is the unique ID for the device.
*
* @return	XST_SUCCESS if successful, otherwise XST_FAILURE.
*
* @note		None.
*
*****************************************************************************/
int SetupTimer(int DeviceID)
{
	int Status;
	XTtcPs_Config *Config;
	XTtcPs *Timer;
	TmrCntrSetup *TimerSetup;

	TimerSetup = &SettingsTable[DeviceID];

	Timer = &(TtcPsInst[DeviceID]);

	/*
	 * Look up the configuration based on the device identifier
	 */
	Config = XTtcPs_LookupConfig(DeviceID);
	if (NULL == Config) {
		return XST_FAILURE;
	}

	/*
	 * Initialize the device
	 */
	Status = XTtcPs_CfgInitialize(Timer, Config, Config->BaseAddress);
	if (Status != XST_SUCCESS) {
		return XST_FAILURE;
	}

	/*
	 * Set the options
	 */
	XTtcPs_SetOptions(Timer, TimerSetup->Options);

	/*
	 * Timer frequency is preset in the TimerSetup structure,
	 * however, the value is not reflected in its other fields, such as
	 * IntervalValue and PrescalerValue. The following call will map the
	 * frequency to the interval and prescaler values.
	 */
	XTtcPs_CalcIntervalFromFreq(Timer, TimerSetup->OutputHz,
		&(TimerSetup->Interval), &(TimerSetup->Prescaler));

	/*
	 * Set the interval and prescale
	 */
	XTtcPs_SetInterval(Timer, TimerSetup->Interval);
	XTtcPs_SetPrescaler(Timer, TimerSetup->Prescaler);

	return XST_SUCCESS;
}

/****************************************************************************/
/**
*
* This function setups the interrupt system such that interrupts can occur.
* This function is application specific since the actual system may or may not
* have an interrupt controller.  The TTC could be directly connected to a
* processor without an interrupt controller.  The user should modify this
* function to fit the application.
*
* @param	IntcDeviceID is the unique ID of the interrupt controller
* @param	IntcInstacePtr is a pointer to the interrupt controller
*		instance.
*
* @return	XST_SUCCESS if successful, otherwise XST_FAILURE.
*
* @note		None.
*
*****************************************************************************/
static int SetupInterruptSystem(u16 IntcDeviceID,
				    XScuGic *IntcInstancePtr)
{
	int Status;
	XScuGic_Config *IntcConfig; /* The configuration parameters of the
					interrupt controller */

	/*
	 * Initialize the interrupt controller driver
	 */
	IntcConfig = XScuGic_LookupConfig(IntcDeviceID);
	if (NULL == IntcConfig) {
		return XST_FAILURE;
	}

	Status = XScuGic_CfgInitialize(IntcInstancePtr, IntcConfig,
					IntcConfig->CpuBaseAddress);
	if (Status != XST_SUCCESS) {
		return XST_FAILURE;
	}

	/*
	 * Connect the interrupt controller interrupt handler to the hardware
	 * interrupt handling logic in the ARM processor.
	 */
	Xil_ExceptionRegisterHandler(XIL_EXCEPTION_ID_INT,
			(Xil_ExceptionHandler) XScuGic_InterruptHandler,
			IntcInstancePtr);

	/*
	 * Enable interrupts in the ARM
	 */
	Xil_ExceptionEnable();

	return XST_SUCCESS;
}

/***************************************************************************/
/**
*
* This function is the handler which handles the periodic tick interrupt.
* It updates its count, and set a flag to signal PWM timer counter to
* update its duty cycle.
*
* This handler provides an example of how to handle data for the TTC and
* is application specific.
*
* @param	CallBackRef contains a callback reference from the driver, in
*		this case it is the instance pointer for the TTC driver.
*
* @return	None.
*
* @note		None.
*
*****************************************************************************/
static void TickHandler(void *CallBackRef, u32 StatusEvent)
{


	Inter_Timer_Intr();


	if (0 != (XTTCPS_IXR_INTERVAL_MASK & StatusEvent)) {
		TickCount++;

		if (TICKS_PER_CHANGE_PERIOD == TickCount) {
			TickCount = 0;
			PWM_UpdateFlag = TRUE;
		}

	}
	else {
		/*
		 * The Interval event should be the only one enabled. If it is
		 * not it is an error
		 */
		ErrorCount++;
	}
}

