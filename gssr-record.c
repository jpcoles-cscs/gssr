/**
 * Collect DCGM metrics for the GPU Saturation Scorer.
 *
 * TODO: 
 *  o Should we quit early if the output path already exists?
 *
 * Written by Jonathan Coles <jonathan.coles@cscs.ch>
 *
 */

#define _GNU_SOURCE
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <time.h>
#include <sys/wait.h>
#include <sys/statfs.h>
//#include <dlfcn.h>

#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <limits.h>

#include "dcgm_agent.h"
#include "dcgm_fields.h"
#include "dcgm_structs.h"
#include "dcgm_errors.h"

#include "gssr-record.h"

/* Metrics */
unsigned short fieldIds[] = {
    DCGM_FI_DEV_GPU_UTIL,
    DCGM_FI_DEV_FB_FREE,
    DCGM_FI_DEV_FB_USED,
    DCGM_FI_DEV_FB_RESERVED,
    DCGM_FI_PROF_SM_ACTIVE,
    DCGM_FI_PROF_SM_OCCUPANCY,
    DCGM_FI_PROF_PIPE_TENSOR_ACTIVE,
    DCGM_FI_PROF_PIPE_FP64_ACTIVE,
    DCGM_FI_PROF_PIPE_FP32_ACTIVE,
    DCGM_FI_PROF_PIPE_FP16_ACTIVE,
    DCGM_FI_PROF_DRAM_ACTIVE,
    DCGM_FI_PROF_PCIE_TX_BYTES,
    DCGM_FI_PROF_PCIE_RX_BYTES,
    DCGM_FI_PROF_NVLINK_RX_BYTES,
    DCGM_FI_PROF_NVLINK_TX_BYTES,
#ifdef DCGM_FI_PROF_C2C_TX_ALL_BYTES
    DCGM_FI_PROF_C2C_TX_ALL_BYTES,
    DCGM_FI_PROF_C2C_RX_ALL_BYTES,
#endif
    DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION,
    DCGM_FI_DEV_POWER_USAGE
};
static int numFields = sizeof(fieldIds) / sizeof(fieldIds[0]);

char *fieldNames[] = {
    "DCGM_FI_DEV_GPU_UTIL",
    "DCGM_FI_DEV_FB_FREE",
    "DCGM_FI_DEV_FB_USED",
    "DCGM_FI_DEV_FB_RESERVED",
    "DCGM_FI_PROF_SM_ACTIVE",
    "DCGM_FI_PROF_SM_OCCUPANCY",
    "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE",
    "DCGM_FI_PROF_PIPE_FP64_ACTIVE",
    "DCGM_FI_PROF_PIPE_FP32_ACTIVE",
    "DCGM_FI_PROF_PIPE_FP16_ACTIVE",
    "DCGM_FI_PROF_DRAM_ACTIVE",
    "DCGM_FI_PROF_PCIE_TX_BYTES",
    "DCGM_FI_PROF_PCIE_RX_BYTES",
    "DCGM_FI_PROF_NVLINK_RX_BYTES",
    "DCGM_FI_PROF_NVLINK_TX_BYTES",
#ifdef DCGM_FI_PROF_C2C_TX_ALL_BYTES
    "DCGM_FI_PROF_C2C_TX_ALL_BYTES",
    "DCGM_FI_PROF_C2C_RX_ALL_BYTES",
#endif
    "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION",
    "DCGM_FI_DEV_POWER_USAGE"
};

#define SAMPLE_INTERVAL         1000000
#define SUBSAMPLE_INTERVAL      (SAMPLE_INTERVAL / 5)
#define RETENTION_INTERVAL      (SAMPLE_INTERVAL + 60)


/* This will be changed by the signal handler so we exit cleanly. */
static volatile int keep_running = 1;
static volatile pid_t child_pid = -1;

static int V = 0;

// ==========================================================================
// handle_sigint - Handle SIGINT and change keep_runing so we exit cleanly.
//
// Returns nothing.
// ==========================================================================
void handle_signal(int sig)
{
    //fprintf(stderr, PROGNAME" got signal %i\n", sig);

    if (child_pid > 0) {
        //fprintf(stderr, PROGNAME" forwarded signal %i to %i\n", sig, child_pid);
        kill(child_pid, sig);   // forward to child
    }

    switch (sig)
    {
        case SIGINT:
        case SIGTERM:
        case SIGQUIT:
        case SIGHUP:
            keep_running = 0;
            break;
    }
}


// ==========================================================================
// install_signal_handler - Register handler for typical signals.
//
// Returns nothing.
// ==========================================================================
void install_signal_handler()
{
    int signals[] = { SIGINT, SIGTERM, SIGQUIT, SIGUSR1, SIGUSR2, SIGHUP };
    int nsignals = sizeof(signals)/sizeof(signals[0]);
    for (int i = 0; i < nsignals; i++) {
        struct sigaction sa = {0};
        sa.sa_handler = handle_signal;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = SA_RESTART;
        sigaction(signals[i], &sa, NULL);
    }
}

// ==========================================================================
// job_environment - Examine envvars and extract SLURM and job related values.
//
// je - the jobenv_t struct to fill in
//
// Returns nothing
// ==========================================================================
void job_environment(jobenv_t *je)
{
    /* Slurm identity */
    je->slurm_rank    = getenv("SLURM_PROCID");
    je->slurm_localid = getenv("SLURM_LOCALID");
    je->slurm_jobname = getenv("SLURM_JOB_NAME");
    je->slurm_jobid   = getenv("SLURM_JOB_ID");
    je->slurm_cluster = getenv("SLURM_CLUSTER_NAME");
    je->slurm_nnodes  = getenv("SLURM_JOB_NUM_NODES");
    je->slurm_ntasks  = getenv("SLURM_NTASKS");
    je->slurm_ngpus   = getenv("SLURM_GPUS_ON_NODE");

    je->slurm_step        = getenv("SLURM_STEP_ID");
    je->slurm_step_nnodes = getenv("SLURM_STEP_NUM_NODES");
    je->slurm_step_ntasks = getenv("SLURM_STEP_NUM_TASKS");

    je->with_slurm = je->slurm_jobname != NULL;

    if (!je->with_slurm) 
    {
        je->slurm_step     = "0";
        je->slurm_rank     = "0";
        je->slurm_localid  = "0";
        je->slurm_jobid    = "0";
        je->slurm_jobname  = "nojobname";
        je->slurm_cluster  = "nocluster";
        je->slurm_ntasks   = "1";
        je->slurm_nnodes   = "1";
        je->slurm_ngpus    = "0";
        je->slurm_step_nnodes = "1";
        je->slurm_step_ntasks = "1";
    }

    je->nnodes = je->slurm_nnodes ? atoi(je->slurm_nnodes) : 0;
    je->ngpus  = je->slurm_ngpus  ? atoi(je->slurm_ngpus)  : 0;
    je->ntasks = je->slurm_ntasks ? atoi(je->slurm_ntasks) : 0;

    je->rank0  = !strncmp("0", je->slurm_rank, strlen(je->slurm_rank));
    je->local0 = !strncmp("0", je->slurm_localid, strlen(je->slurm_localid));
}

// ==========================================================================
// parse_args - Parse the commandline arguments
//
// argc - size of argv
// argv - list of arguments to parse
// args - the cmdargs_t struct to complete
//
// Returns nothing
// ==========================================================================
void parse_args(int argc, char **argv, cmdargs_t *args)
{
    assert(argc >= 1);

    memset(args, 0, sizeof(*args));

    int i;
    for (i=1; i < argc; i++)
    {
        if (argv[i][0] != '-')
        {
            break;
        }
        else
        {
            if (!strcmp("-h", argv[i]) || !strcmp("--help", argv[i]))
            {
                args->show_help = 1;
            }
            else if (!strcmp("-v", argv[i]))
            {
                args->verbose += 1;
            }
            else if (!strcmp("-vv", argv[i]))
            {
                args->verbose += 2;
            }
            else if (!strcmp("--version", argv[i]))
            {
                args->show_version = 1;
            }
            else if (!strcmp("-o", argv[i]))
            {
                if (i+1 < argc)
                {
                    i++;
                    if (!strcmp("--", argv[i])) 
                    {
                        fprintf(stderr, PROGNAME": Missing directory argument to -o\n");
                        args->show_help = 1;
                    }
                    else
                    {
                        args->outdir = argv[i];
                    }
                }
                else
                {
                    args->show_help = 1;
                }
            }
            else if (!strcmp("--test-only", argv[i]))
            {
                args->test_only = 1;
            }
            else if (!strcmp("--", argv[i])) 
            {
                i++;
                break;
            }
            else
            {
                //fprintf(stder, PROGNAME": Unrecognized argument %s. Use -- to signal end of arguments to %s.\n", argv[i], PROGNAME);
                args->show_help = 1;
            }
        }
    }

    if (args->show_help) return;

    args->child_argc = argc - i;
    /* one extra for terminating NULL needed by execvp */
    args->child_argv = (char **)malloc(sizeof(char *) * (args->child_argc+1));
    if (!args->child_argv)
    {
        perror("Failed to allocate memory for the child process");
        exit(1);
    }

    for (int j=0; i < argc && j < args->child_argc; i++, j++)
    {
        args->child_argv[j] = argv[i];
    }
    /* We need this for execvp later */
    args->child_argv[args->child_argc] = NULL;
}

// ==========================================================================
// record_metrics - Send a request to DCGM for field summaries
//
// handle - DCGM handle
// records - record_t array to append to with latest results
// record_count - current number of items in records. Will be updated.
// timestamp - timestamp to associate with new records
// then, now - time window for DCGM request (in seconds)
// devices - list of devices to measure
// numDevices - size of devices array
//
// Returns nothing
// ==========================================================================
void record_metrics(dcgmHandle_t handle, 
                    record_t *records, 
                    int *record_count, 
                    time_t timestamp, 
                    time_t then, time_t now, 
                    unsigned int *devices, int numDevices)
{
    dcgmFieldSummaryRequest_t req = {0};

    req.version = dcgmFieldSummaryRequest_version1;
    req.entityGroupId = DCGM_FE_GPU;

    /* Requests need to be in usec */
    req.startTime = then * 1000000;
    req.endTime   = now  * 1000000;

    for (int gpu_idx = 0; gpu_idx < numDevices; gpu_idx++)
    {
        records[*record_count]       = (record_t){0};
        records[*record_count].ts    = timestamp;
        records[*record_count].gpuId = devices[gpu_idx];

        for (int field_idx=0; field_idx < numFields; field_idx++)
        {
            req.entityId = devices[gpu_idx];
            req.fieldId  = fieldIds[field_idx];
            req.summaryTypeMask =
                DCGM_SUMMARY_MIN |
                DCGM_SUMMARY_MAX |
                DCGM_SUMMARY_AVG;

            if (fieldIds[field_idx] == DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION)
            {
                req.summaryTypeMask =
                    DCGM_SUMMARY_MIN |
                    DCGM_SUMMARY_MAX |
                    DCGM_SUMMARY_DIFF;
            }

            if (dcgmGetFieldSummary(handle, &req) != DCGM_ST_OK) {
                //fprintf(stderr, PROGNAME": Error getting %s\n", fieldNames[field_idx]);
                goto skip_record;
            }

            if (req.response.summaryCount != 3) {
                fprintf(stderr, PROGNAME": Incorrect summary count from dcgmGetFieldSummary. Expected 3, got %i. Trying to continue.\n", req.response.summaryCount);
                goto skip_record;
            }

            records[*record_count].values[field_idx].fieldId = fieldIds[field_idx];
            records[*record_count].values[field_idx].fieldType = req.response.fieldType;

            if (req.response.fieldType == DCGM_FT_INT64)
            {
                records[*record_count].values[field_idx].min = req.response.values[0].i64 > 2000000000 ? 0 : req.response.values[0].i64;
                records[*record_count].values[field_idx].max = req.response.values[1].i64 > 2000000000 ? 0 : req.response.values[1].i64;
                records[*record_count].values[field_idx].avg = req.response.values[2].i64 > 2000000000 ? 0 : req.response.values[2].i64;
            }
            else
            {
                records[*record_count].values[field_idx].min = req.response.values[0].fp64 > 2e9 ? 0 : req.response.values[0].fp64;
                records[*record_count].values[field_idx].max = req.response.values[1].fp64 > 2e9 ? 0 : req.response.values[1].fp64;
                records[*record_count].values[field_idx].avg = req.response.values[2].fp64 > 2e9 ? 0 : req.response.values[2].fp64;
            }
        }

        (*record_count)++;
skip_record:
        ;
    }
}

// ==========================================================================
// write_meta - Write meta data
//
// fp - pointer to open file
// jobenv - jobenv_t containing the meta data to save
//
// Returns nothing
// ==========================================================================
void write_meta(FILE *fp, cmdargs_t *args, jobenv_t *jobenv)
{
    char date[32];  // "YYYY-MM-DDTHH:MM:SS..." + '\0'

    time_t now = time(NULL);
    struct tm *tm_now = localtime(&now);
    strftime(date, sizeof(date), "%Y-%m-%dT%H:%M:%S%z", tm_now);

    fprintf(fp,
        "{\n"
        "    \""PROGNAME"-version\": \""VERSION"\""      ",\n"
        "    \"date\": \"%s\""       ",\n"
        "    \"cluster\": \"%s\""    ",\n"
        "    \"jobid\": \"%s\""      ",\n"
        "    \"jobname\": \"%s\""    ",\n"
        "    \"nnodes\": \"%s\""     ",\n"
        "    \"ntasks\": \"%s\""     ",\n"
        "    \"ngpus\": \"%s\""       ",\n"
        "    \"step_nnodes\": \"%s\""     ",\n"
        "    \"step_ntasks\": \"%s\""     ",\n"
        "    \"executable\": \"%s\"" ",\n",
        date,
        jobenv->slurm_cluster,
        jobenv->slurm_jobid,
        jobenv->slurm_jobname,
        jobenv->slurm_nnodes,
        jobenv->slurm_ntasks,
        jobenv->slurm_ngpus,
        jobenv->slurm_step_nnodes,
        jobenv->slurm_step_ntasks,
        args->child_argv[0]
    );

    fprintf(fp, "    \"arguments\": \"");
    for (int i=1; i < args->child_argc; i++)
    {
        fprintf(fp, "'%s' ", args->child_argv[i]);
    }
    fprintf(fp, "\""  "\n");

    fprintf(fp, "}\n\n");
}

// ==========================================================================
// write_records - Write records to file in CSV format
//
// fp - pointer to open file
// n - number of records to write
// with_header - optionally write a header with column names.
//
// with_header can be activated for the first call to write_records and
// then left at 0 for subsequent calls.
//
// Returns nothing
// ==========================================================================
void write_records(FILE *fp, int n, record_t *records, int with_header)
{
    assert(fp != NULL);
    assert(records != NULL);

    if (with_header) 
    {
        fprintf(fp, "timestamp,gpuId");
        for (int field_idx=0; field_idx < numFields; field_idx++)
        {
            fprintf(fp, ",%s_min", fieldNames[field_idx]);
            fprintf(fp, ",%s_avg", fieldNames[field_idx]);
            fprintf(fp, ",%s_max", fieldNames[field_idx]);
        }
        fprintf(fp, "\n");
    }

    for (int i = 0; i < n; ++i) 
    {
        fprintf(fp, "%ld,%u", records[i].ts, records[i].gpuId);
        for (int j = 0; j < numFields; j++) 
        {
            if (records[i].values[j].fieldType == DCGM_FT_INT64)
            {
                fprintf(fp, ",%ld,%ld,%ld",
                        (int64_t)records[i].values[j].min,
                        (int64_t)records[i].values[j].avg,
                        (int64_t)records[i].values[j].max);
            }
            else
            {
                fprintf(fp, ",%.2f,%.2f,%.2f",
                        records[i].values[j].min,
                        records[i].values[j].avg,
                        records[i].values[j].max);
            }
        }
        fprintf(fp, "\n");
    }

    fflush(fp);
}

// ==========================================================================
// child_finished - Determine if forked child process has finished
//
// child_pid - The PID from fork
//
// OLD: This function currently doesn't use the child PID. It more generally
// determines whether all child (and possibly grandchild) processes have
// finished.
//
// Returns 1 if all child processes have exited, 0 otherwise.
// ==========================================================================
int child_finished(pid_t child_pid)
{
    int status;
    pid_t pid = waitpid(child_pid, &status, WNOHANG);
    //pid_t pid = waitpid(-1, &status, WNOHANG);

    if (pid == -1)
    {
        //if (errno != EINTR)
            //return 1;
        if (errno == ECHILD)
            return 1;
    }
    else if (pid > 0)
    {
        if (WIFEXITED(status) && WEXITSTATUS(status) != 0) 
        {
            fprintf(stderr, PROGNAME" warning: Workload returned non-zero exit code: %i\n", WEXITSTATUS(status));
        } 
        else if (WIFSIGNALED(status)) 
        {
            fprintf(stderr, PROGNAME" warning: Workload was terminated by signal: %i\n", WTERMSIG(status));
        }
    }

    return 0;
}

// ==========================================================================
// version - Print version information to stdout
//
// This function doesn't exit the program.
//
// Returns nothing
// ==========================================================================
void version()
{
    printf(PROGNAME" "VERSION);
}

// ==========================================================================
// help - Print usage information to stdout
//
// This function doesn't exit the program.
//
// Returns nothing
// ==========================================================================
void help()
{
    printf(
        "Usage: "PROGNAME" [OPTIONS] <cmd> [args...]\n"
        "Run cmd and record GPU metrics. Results can be given to the\n" 
        "GPU saturation scorer (GSSR) to produce a report for CSCS project proposals.\n"
        "\n"
        "   -h | --help         Display this help message.\n"
        "   --version           Show version information.\n"
        "   -o <directory>      Create given directory and write results there.\n"
        "\n"
        PROGNAME" depends on the NVIDIA DCGM library. When running in a\n"
        "container at CSCS you will need the Container Engine annotation in\n"
        "your container EDF file:\n"
        "\n"
        "[annotations]\n"
        "com.hooks.dcgm.enabled = \"true\"\n"
        "\n\n"
        "Report issues to CSCS Service Desk  https://support.cscs.ch\n"
        "\n"
    );
}

// ==========================================================================
// mkdir_p - Create all non-existing directories in the given path
//
// path - Path to create
// mode - permissions to use (700 recommended)
//
// Returns 0 on success, otherwisee -1 and sets errno
// ==========================================================================
int mkdir_p(const char *path, mode_t mode)
{
    char tmp[PATH_MAX];
    char *p = NULL;
    size_t len;
    struct stat st;

    if (!path || !*path)
        return -1;

    len = strnlen(path, sizeof(tmp));
    if (len >= sizeof(tmp)) {
        errno = ENAMETOOLONG;
        return -1;
    }

    memcpy(tmp, path, len + 1);

    /* Remove trailing slash (except for root) */
    if (len > 1 && tmp[len - 1] == '/')
        tmp[len - 1] = '\0';

    for (p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';

            if (mkdir(tmp, mode) != 0) {
                if (errno != EEXIST)
                    return -1;

                if (stat(tmp, &st) != 0 || !S_ISDIR(st.st_mode)) {
                    errno = ENOTDIR;
                    return -1;
                }
            }

            *p = '/';
        }
    }

    /* Final component */
    if (mkdir(tmp, mode) != 0) {
        if (errno != EEXIST)
            return -1;

        if (stat(tmp, &st) != 0 || !S_ISDIR(st.st_mode)) {
            errno = ENOTDIR;
            return -1;
        }
    }

    return 0;
}

void warnfs(char *msg, __fsword_t fstype)
{
    switch (fstype)
    {
        case 0x65735546: /* FUSE_SUPER_MAGIC */
            fprintf(stderr, PROGNAME": %s may be inside a container. Any data recorded there will be lost.\n", msg);
            fprintf(stderr, PROGNAME": Bind-mount an external path and save data there.\n");
            break;
        default:
            break;
    }
}

// ==========================================================================
// create_output_location - Create the appropriate path and output file
//
// Attempt to create "report_<cluster>_<jobid>/step_<step>/proc_<procid>.csv" which
// may require making the parent and sub-directories first.
//
// csvfp - Updated with the opened records file
// metafp - Updated with the opened metadata file. Set to NULL to ignore.
// fp - pointer to a file pointer that will be updated with the opened file
// jobenv - jobenv_t struct describing job environment
// args - cmdargs_t describing the command line arguments
//
// Returns 1 on success and 0 otherwise.
// ==========================================================================
int create_output_location(FILE **csvfp, FILE **metafp, jobenv_t *jobenv, cmdargs_t *args)
{
    int  ret = 1;
    char *dirname;
    char *fname;
    char *metafname;

    char *cwd = (char *)malloc(PATH_MAX * sizeof(*cwd));  // PATH_MAX is the maximum path length
    if (getcwd(cwd, PATH_MAX) != NULL) 
    {
        if (V >= 2) fprintf(stderr, PROGNAME": Current directory %s.\n", cwd);
    }
    if (cwd) free(cwd);

    if (V >= 2) fprintf(stderr, PROGNAME": Creating output locations.\n");

    if (args->outdir)
    {
        struct statfs buf;
        statfs(args->outdir, &buf);
        warnfs(args->outdir, buf.f_type);
    }
    else
    {
        struct statfs buf;
        statfs(".", &buf);
        warnfs("The current directory", buf.f_type);
    }


    if (args->outdir)
    {
        ret = asprintf(&dirname, "%s/%s_%s/step_%s", args->outdir, jobenv->slurm_cluster, jobenv->slurm_jobid, jobenv->slurm_step);
    }
    else
    {
        ret = asprintf(&dirname, "gssr_report/%s_%s/step_%s", jobenv->slurm_cluster, jobenv->slurm_jobid, jobenv->slurm_step);
    }
    if (ret < 0)
    {
        fprintf(stderr, PROGNAME": failed to allocate memory. Exiting.\n");
        goto cleanup;
    }
    ret = asprintf(&fname, "%s/proc_%s.csv", dirname, jobenv->slurm_rank);
    if (ret < 0)
    {
        fprintf(stderr, PROGNAME": failed to allocate memory. Exiting.\n");
        goto cleanup;
    }
    ret = asprintf(&metafname, "%s/proc_%s.meta.txt", dirname, jobenv->slurm_rank);
    if (ret < 0)
    {
        fprintf(stderr, PROGNAME": failed to allocate memory. Exiting.\n");
        goto cleanup;
    }

    if (V >= 2) fprintf(stderr, PROGNAME": Making directory %s.\n", dirname);
    if (mkdir_p(dirname, S_IRWXU))
    {
        if (jobenv->rank0) fprintf(stderr, PROGNAME": Cannot create output directory %s\n", dirname);
        ret = -1;
        goto cleanup;
    }

    if (V >= 2) fprintf(stderr, PROGNAME": Making csv file %s.\n", fname);
    *csvfp = fopen(fname, "wt");

    if (!*csvfp)
    {
        fprintf(stderr, PROGNAME": Cannot create output file %s\n", fname);
        ret = -1;
        goto cleanup;
    }

    if (metafp != NULL)
    {
        if (V >= 2) fprintf(stderr, PROGNAME": Making meta file %s.\n", metafname);
        *metafp = fopen(metafname, "wt");
        if (!*metafp)
        {
            fprintf(stderr, PROGNAME": Cannot create meta file %s\n", metafname);
            ret = -1;
            goto cleanup;
        }
    }

    ret = 1;

cleanup:
    free(dirname);
    free(fname);
    free(metafname);

    return ret > 0;
}

void make_args_coherent(cmdargs_t *args)
{
    if (args->show_version)
        args->show_help = 0;

    if (args->show_help)    return;
    if (args->show_version) return;
    if (args->test_only)    return;

    if (args->child_argc == 0)
        args->show_help = 1;

    V = args->verbose;
}

// ==========================================================================
// main
// ==========================================================================
int main(int argc, char **argv)
{
    int main_ret = EXIT_FAILURE;
    FILE *fp = NULL;
    FILE *metafp = NULL;
    jobenv_t jobenv;
    cmdargs_t args;

    // ------------------------------------------------------------------------
    //  Extract parameters from the environmen and command line arguments.
    //
    //  Handle simple cases like help and no child process command.
    // ------------------------------------------------------------------------

    job_environment(&jobenv);
    parse_args(argc, argv, &args);

    make_args_coherent(&args);

    if (args.show_help)
    {
        if (jobenv.rank0)
        {
            help();
        }
        goto stop;
    }

    if (args.show_version)
    {
        if (jobenv.rank0)
        {
            version();
        }
        goto stop;
    }

    if (args.test_only)
    {
        if (jobenv.rank0)
        {
            run_tests();
        }
        goto stop;
    }

    if (args.child_argc == 0)
    {
        if (jobenv.rank0) fprintf(stderr, PROGNAME": No command given.\n");
        goto stop;
    }

    if (!jobenv.with_slurm)
    {
        if (jobenv.rank0) fprintf(stderr, PROGNAME": Not running inside a Slurm step\n");
        if (jobenv.rank0) fprintf(stderr, PROGNAME": Results may be affected by other jobs running on the system.\n");
    }


//  void *h = dlopen("libdcgm.so", RTLD_NOW);
//  if (!h) {
//      fprintf(stderr,
//          "libdcgm.so not found.\n"
//          "Hint: load module foo/2.3 or set LD_LIBRARY_PATH.\n"
//          "dlerror: %s\n", dlerror());
//      exit(1);
//  }

    // ------------------------------------------------------------------------
    // If we are not the local root rank just start the child process.
    // ------------------------------------------------------------------------
    if (!jobenv.local0)
    {
        if (V >= 2) fprintf(stderr, PROGNAME": Forking child.\n");
        // Child process
        execvp(args.child_argv[0], args.child_argv);
        /* No message here, rank0 will report */
        goto stop;
    }

    // ------------------------------------------------------------------------
    // Create the directory structure and output files.
    //
    // Best to fail here before we fork.
    // ------------------------------------------------------------------------
    if (!create_output_location(
                &fp, 
                jobenv.rank0 ? &metafp : NULL, 
                &jobenv, &args))
    {
        /* No message, create_output_location will do that. */
        goto close_files;
    }

    // ------------------------------------------------------------------------
    //
    // Fork a child process and launch the program to profile.
    //
    // ------------------------------------------------------------------------

    struct sigaction sa;
    sa.sa_handler = SIG_IGN;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_NOCLDWAIT;
    sigaction(SIGCHLD, &sa, NULL);

    child_pid = fork();
    if (child_pid == 0) 
    {
        // Child process
        if (V >= 2) fprintf(stderr, PROGNAME": Starting %s.\n", args.child_argv[0]);
        execvp(args.child_argv[0], args.child_argv);
        // If execvp returns, there was an error
        if (jobenv.rank0) perror("Failed to execute the command. Is it in the path?");
        goto reset_sighandler;
    } 
    else if (child_pid < 0)
    {
        // Fork failed
        if (jobenv.rank0) perror("Failed to fork process. Not enough system resources?");
        goto reset_sighandler;
    }


    if (V >= 2) fprintf(stderr, PROGNAME": Setting up DCGM connection.\n");

    // ------------------------------------------------------------------------
    // Install the signal handler that will cause our monitoring loop to exit.
    // It will also forward signals to the child that might be coming from 
    // slurm.
    // ------------------------------------------------------------------------
    install_signal_handler();
    
    // ------------------------------------------------------------------------
    // Now that we are safely past the fork we can start setting up DCGM.
    // ------------------------------------------------------------------------

    /* Connect to hostengine (standalone mode) */
    dcgmHandle_t handle;
    if (dcgmInit() != DCGM_ST_OK)
    {
        fprintf(stderr, PROGNAME": Unable to initialize DCGM. Command will not be monitored.\n");
        goto reset_sighandler;
    }
    if (dcgmConnect("127.0.0.1", &handle) != DCGM_ST_OK)
    {
        fprintf(stderr, PROGNAME": Unable to connect to DCGM host engine at 127.0.0.1. Command will not be monitored.\n");
        goto shutdown;
    }

    char gpu_group_name[10+1+7+1];
    char field_group_name[7+1+7+1];

    snprintf(  gpu_group_name, sizeof(  gpu_group_name), "slurm_gpus_%ld", getpid());
    snprintf(field_group_name, sizeof(field_group_name), "metrics_%ld", getpid());

    /* Create GPU group */
    dcgmGpuGrp_t gpuGroup;
    if (dcgmGroupCreate(handle, DCGM_GROUP_EMPTY, gpu_group_name, &gpuGroup) != DCGM_ST_OK)
    {
        fprintf(stderr, PROGNAME": Unable to create a DCGM monitoring group. Command will not be monitored.\n");
        goto disconnect;
    }

    /* Field group */
    dcgmFieldGrp_t fieldGroup;
    if (dcgmFieldGroupCreate(handle, numFields, fieldIds, field_group_name, &fieldGroup) != DCGM_ST_OK)
    {
        fprintf(stderr, PROGNAME": Unable to create a DCGM monitoring field group. Command will not be monitored.\n");
        goto destroy_fieldgroup;
    }

    unsigned int devices[MAX_GPUS];
    int numDevices;

    if (dcgmGetAllSupportedDevices(handle, devices, &numDevices) != DCGM_ST_OK)
    {
        fprintf(stderr, PROGNAME": Unable determine supported devices. Command will not be monitored.\n");
        goto destroy_fieldgroup;
    }

    jobenv.ngpus = numDevices;

    for (int i = 0; i < numDevices; i++) 
    {
        if (dcgmGroupAddDevice(handle, gpuGroup, devices[i]) != DCGM_ST_OK)
        {
            fprintf(stderr, PROGNAME": Unable add device %i to be monitored. Command will not be monitored.\n", i);
            goto destroy_fieldgroup;
        }
    }

    if (dcgmWatchFields(handle, gpuGroup, fieldGroup, SUBSAMPLE_INTERVAL, RETENTION_INTERVAL, 0) != DCGM_ST_OK)
    {
        fprintf(stderr, PROGNAME": Unable monitor the requested fields. Command will not be monitored.\n");
        goto destroy_fieldgroup;
    }

    if (dcgmUpdateAllFields(handle, 1) != DCGM_ST_OK)
    {
        fprintf(stderr, PROGNAME": Unable monitor the requested fields. Command will not be monitored.\n");
        goto unwatch;
    }

    // ------------------------------------------------------------------------
    // Begin the measuring loop. Keep going until the child process has ended.
    // ------------------------------------------------------------------------

    if (jobenv.rank0)
    {
        if (metafp)
        {
            if (V >= 2) fprintf(stderr, PROGNAME": Writing metadata.\n");
            write_meta(metafp, &args, &jobenv);
            fflush(metafp);
            fclose(metafp);
            metafp = NULL;
        }
    }

    int    write_header=1;
    time_t start = time(NULL);
    time_t then  = start;

    record_t *records = (record_t *)malloc(MAX_RECORDS * sizeof(*records));
    if (!records)
    {
        fprintf(stderr, PROGNAME": failed to allocation memory for records. Exiting.\n");
        goto cleanup;
    }
    int record_count = 0;

    if (V >= 2) fprintf(stderr, PROGNAME": Starting monitoring loop.\n");
    while (keep_running && record_count < MAX_RECORDS) 
    {
        if (child_finished(child_pid))
            break;

        usleep(SAMPLE_INTERVAL);

        time_t now = time(NULL);

        record_metrics(handle, records, &record_count, now-start, then, now, devices, numDevices);

        then = now;

        if (record_count > WRITE_RECORDS_TRIGGER)
        {
            write_records(fp, record_count, records, write_header);
            write_header = 0;
            record_count = 0;
        }
    }

    write_records(fp, record_count, records, write_header);

    time_t stop = time(NULL);

    if (stop - start < APP_RUNNING_TIME_WARNING)
    {
        if (jobenv.rank0) fprintf(stderr, PROGNAME" warning: Job lasted less than %i "
                "seconds. This may be too short to record meaningful data.\n",
                APP_RUNNING_TIME_WARNING); 
    }

    if (V >= 2) fprintf(stderr, PROGNAME": Waiting for children.\n");

    /* If a signal came in that the child handles it may need time to clean up. */
    while (!child_finished(child_pid)) {}

    // ------------------------------------------------------------------------
    // Shutdown DCGM and cleanup
    // ------------------------------------------------------------------------

    if (V >= 2) fprintf(stderr, PROGNAME": Made it to shutdown.\n");
    main_ret = EXIT_SUCCESS;

cleanup:
    ;

unwatch:
    dcgmUnwatchFields(handle, gpuGroup, fieldGroup);
destroy_fieldgroup:
    dcgmFieldGroupDestroy(handle, fieldGroup);
destroy_group:
    dcgmGroupDestroy(handle, gpuGroup);
disconnect:
    dcgmDisconnect(handle);
shutdown:
    dcgmShutdown();

reset_sighandler:

    // Reset signal handlers to default
    sa.sa_handler = SIG_DFL;
    sigaction(SIGCHLD, &sa, NULL);

close_files:

    if (fp) 
    {
        fflush(fp);
        fclose(fp);
    }

    if (metafp) 
    {
        fflush(metafp);
        fclose(metafp);
    }

stop:

    if (V >= 2) fprintf(stderr, PROGNAME": Exiting.\n");

    return main_ret;
}

