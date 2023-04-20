#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <sys/mman.h>
#include <sys/timerfd.h>
#include <sys/types.h>


#define SHARED_RAM_POS_R0toR1 0x70000000
#define SHARED_RAM_POS_R1toR0 0x70100000
#define SHARED_RAM_POS_AtoR0 0x70200000
#define SHARED_RAM_POS_R0toA 0x70300000
#define SHARED_RAM_POS_AtoR1 0x70400000
#define SHARED_RAM_POS_R1toA 0x70500000


#define SHARED_RAM_LENGTH 0x100000

#define handle_error(msg) do { perror(msg); exit(-1); } while (0)

unsigned int Data_Test[4]={1,2,3,4};

int main(){
    int dh = open("/dev/mem", O_RDWR | O_SYNC); // Open /dev/mem which represents the whole physical memory
    unsigned int* RtnR0 = (unsigned int *)mmap(NULL, SHARED_RAM_LENGTH, PROT_READ | PROT_WRITE, MAP_SHARED, dh, SHARED_RAM_POS_R0toA);
    unsigned int* RtnR1 = (unsigned int *)mmap(NULL, SHARED_RAM_LENGTH, PROT_READ | PROT_WRITE, MAP_SHARED, dh, SHARED_RAM_POS_R1toA);
    unsigned int* ToR0 = (unsigned int *)mmap(NULL, SHARED_RAM_LENGTH, PROT_READ | PROT_WRITE, MAP_SHARED, dh, SHARED_RAM_POS_AtoR0);
    unsigned int* ToR1 = (unsigned int *)mmap(NULL, SHARED_RAM_LENGTH, PROT_READ | PROT_WRITE, MAP_SHARED, dh, SHARED_RAM_POS_AtoR1);
    unsigned int i;


    struct itimerspec new_value;
    int fd, exp;
    struct timespec now;
    ssize_t s;
    if (clock_gettime(CLOCK_REALTIME, &now) == -1){
        handle_error("clock_gettime");
    }
    new_value.it_value.tv_sec = now.tv_sec + 1;
    new_value.it_value.tv_nsec = now.tv_nsec;
    new_value.it_interval.tv_sec = 0;
    new_value.it_interval.tv_nsec = 1666666;

    fd = timerfd_create(CLOCK_REALTIME, 0);
    if (fd == -1) {
        handle_error("timerfd_create");
    }

    if (timerfd_settime(fd, TFD_TIMER_ABSTIME, &new_value, NULL) == -1) {
        handle_error("timerfd_settime");
    }

    int counter = 0;

    char filename[25];

    time_t ct;
    time(&ct);
    struct tm *p;
    p = gmtime(&ct);
    sprintf(filename,"f_%d%d%d_%d%d%d", (1900+p->tm_year), (1+p->tm_mon), p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);

    FILE *cfptr;
    cfptr = fopen(filename, "w");



    while(1){
        //
        // reading timer
        //
        s = read(fd, &exp, sizeof(u_int64_t));
        if (s != sizeof(u_int64_t)) {
            handle_error("read");
        }

    	Data_Test[0]++;
    	Data_Test[1]++;
    	Data_Test[2]++;
    	Data_Test[3]++;

    	char cmd[256];
    	sprintf(cmd, "%d\t%d\t%d\t%d\n", Data_Test[0], Data_Test[1], Data_Test[2], Data_Test[3]);
    	fprintf(cfptr, "%s", cmd);
        
        //memcpy( ToR0, RtnR0, 512 );
        //memcpy( ToR1, RtnR1, 512 );

        ToR0[0] = RtnR0[0];
        ToR1[0] = RtnR1[0];
	if (counter == 199){
            printf("1110930 Run APU 200 ticks reached. toR0: %d toR1: %d\n", ToR0[0], ToR1[0]);
            //printf("Run APU 600 ticks reached. \n");
	    counter = 0;
	}else counter ++;
    }

    close(dh);
    return 0;
}

