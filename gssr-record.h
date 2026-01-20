#ifndef GSSR_RECORD_H

#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/stat.h>

#include "dcgm_agent.h"
#include "dcgm_fields.h"
#include "dcgm_structs.h"
#include "dcgm_errors.h"

#define PROGNAME "gssr-record"
#define VERSION "2.0"

/* Maximum number of fields to watch */
#define MAX_FIELDS  20
/* Maximum number of records to store. Normally we write our results before we get here. */
#define MAX_RECORDS 200
/* Limit on how many GPUs we handle. Even with MIG on Alps we shouldn't get this high. */
#define MAX_GPUS    32
/* Write records to disk after storing at least this many. Should be much less than MAX_RECORDS. */
/* Since this will only be checked in the polling loop, it's proportional to the number
 * of seconds the program has been running. */
#define WRITE_RECORDS_TRIGGER 40
/* Warn the user if the application ran for less than this many seconds. Not enough
 * meaningful data may have been collected. */
#define APP_RUNNING_TIME_WARNING 10

#define CHECK_DCGM(call)                                      \
    do {                                                      \
        dcgmReturn_t _ret = (call);                           \
        if (_ret != DCGM_ST_OK) {                             \
            fprintf(stderr,                                  \
                    PROGNAME" %s:%i] DCGM error %d (%s)\n",                   \
                    __FILE__, __LINE__, _ret, errorString(_ret));             \
            exit(EXIT_FAILURE);                               \
        }                                                     \
    } while (0)

typedef struct {
    unsigned short fieldId;
    int fieldType;
    double min, avg, max;
} field_summary_t;

typedef struct {
    time_t ts;
    unsigned int gpuId;
    field_summary_t values[MAX_FIELDS];
} record_t;

/*
 * Captures the job environment and has a few logical variables
 * to help later decision making.
 */
typedef struct
{
    char *slurm_step;
    char *slurm_rank;
    char *slurm_localid;
    char *slurm_jobid;
    char *slurm_jobname;
    char *slurm_cluster;
    char *slurm_nnodes;
    char *slurm_ntasks;
    char *slurm_ngpus;
    char *slurm_step_nnodes;
    char *slurm_step_ntasks;
    int rank0;
    int local0;
    int ntasks;
    int with_slurm;
    int ngpus;
    int nnodes;
} jobenv_t;

/*
 * Parsed command line arguments including the child command.
 */
typedef struct
{
    int show_help;
    int show_version;
    char *outdir;
    char **child_argv;
    int  child_argc;
    int test_only;
} cmdargs_t;


void version();
void help();
void write_meta(FILE *fp, cmdargs_t *args, jobenv_t *jobenv);
int create_output_location(FILE **csvfp, FILE **metafp, jobenv_t *jobenv, cmdargs_t *args);
void parse_args(int argc, char **argv, cmdargs_t *args);
void job_environment(jobenv_t *je);
void run_tests();

#endif
