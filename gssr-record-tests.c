#include <assert.h>
#include <string.h>
#include "gssr-record.h"

void test_job_environment()
{
    jobenv_t je;

    setenv("SLURM_STEP_ID", "123", 1);
    job_environment(&je);
    assert(!strcmp(je.slurm_step, "0"));
    assert(!strcmp(je.slurm_rank, "0"));
    assert(!strcmp(je.slurm_localid, "0"));
    assert(!strcmp(je.slurm_jobid, "0"));
    assert(!strcmp(je.slurm_jobname, "nojobname"));
    unsetenv("SLURM_STEP_ID");

    setenv("SLURM_PROC_ID", "123", 1);
    job_environment(&je);
    assert(!strcmp(je.slurm_step, "0"));
    assert(!strcmp(je.slurm_rank, "0"));
    assert(!strcmp(je.slurm_localid, "0"));
    assert(!strcmp(je.slurm_jobid, "0"));
    assert(!strcmp(je.slurm_jobname, "nojobname"));
    unsetenv("SLURM_PROC_ID");

    setenv("SLURM_LOCALID", "123", 1);
    job_environment(&je);
    assert(!strcmp(je.slurm_step, "0"));
    assert(!strcmp(je.slurm_rank, "0"));
    assert(!strcmp(je.slurm_localid, "0"));
    assert(!strcmp(je.slurm_jobid, "0"));
    assert(!strcmp(je.slurm_jobname, "nojobname"));
    unsetenv("SLURM_LOCALID");

    setenv("SLURM_JOB_NAME", "test", 1);
    setenv("SLURM_JOB_ID",   "000", 1);
    setenv("SLURM_STEP_ID",  "123", 1);
    setenv("SLURM_PROCID",   "456", 1);
    setenv("SLURM_LOCALID",  "789", 1);
    job_environment(&je);
    assert(!strcmp(je.slurm_step, "123"));
    assert(!strcmp(je.slurm_rank, "456"));
    assert(!strcmp(je.slurm_localid, "789"));
    assert(!strcmp(je.slurm_jobid, "000"));
    assert(!strcmp(je.slurm_jobname, "test"));
    assert(!je.rank0);
    assert(!je.local0);

    setenv("SLURM_JOB_NAME", "test", 1);
    setenv("SLURM_JOB_ID",   "000", 1);
    setenv("SLURM_STEP_ID",  "123", 1);
    setenv("SLURM_PROCID",   "0", 1);
    setenv("SLURM_LOCALID",  "0", 1);
    job_environment(&je);
    assert(!strcmp(je.slurm_step, "123"));
    assert(!strcmp(je.slurm_rank, "0"));
    assert(!strcmp(je.slurm_localid, "0"));
    assert(!strcmp(je.slurm_jobid, "000"));
    assert(!strcmp(je.slurm_jobname, "test"));
    assert(je.rank0);
    assert(je.local0);

    setenv("SLURM_CLUSTER_NAME", "test-cluster", 1);
    setenv("SLURM_JOB_NAME", "test", 1);
    setenv("SLURM_JOB_ID",   "000", 1);
    setenv("SLURM_STEP_ID",  "123", 1);
    setenv("SLURM_PROCID",   "0", 1);
    setenv("SLURM_LOCALID",  "0", 1);
    setenv("SLURM_NTASKS",   "5", 1);
    setenv("SLURM_JOB_NUM_NODES",   "8", 1);
    job_environment(&je);
    assert(!strcmp(je.slurm_cluster, "test-cluster"));
    assert(!strcmp(je.slurm_step, "123"));
    assert(!strcmp(je.slurm_rank, "0"));
    assert(!strcmp(je.slurm_localid, "0"));
    assert(!strcmp(je.slurm_jobid, "000"));
    assert(!strcmp(je.slurm_jobname, "test"));
    assert(!strcmp(je.slurm_ntasks, "5"));
    assert(je.ntasks == 5);
    assert(!strcmp(je.slurm_nnodes, "8"));
    assert(je.nnodes == 8);
    assert(je.rank0);
    assert(je.local0);
}

#define ASSERT_CLEAN(cond, label) \
    if (!(cond)) { fprintf(stderr, "Assert failed on %s:%i\n", __FILE__, __LINE__); goto label;}

void test_parse_cuda_visible_devices()
{
    {
    char *str = "0,1,2,3";
    unsigned int visible[MAX_GPUS] = {0};
    int numVisible = 0;
    numVisible = parse_cuda_visible_devices(str, visible, MAX_GPUS);
    assert(numVisible == 4);
    assert(visible[0] == 0);
    assert(visible[1] == 1);
    assert(visible[2] == 2);
    assert(visible[3] == 3);
    }

    {
    char *str = "0";
    unsigned int visible[MAX_GPUS] = {0};
    int numVisible = 0;
    numVisible = parse_cuda_visible_devices(str, visible, MAX_GPUS);
    assert(numVisible == 1);
    }

    {
    char *str = "";
    unsigned int visible[MAX_GPUS] = {0};
    int numVisible = 10;
    numVisible = parse_cuda_visible_devices(str, visible, MAX_GPUS);
    assert(numVisible == 0);
    }

    {
    char *str = ",";
    unsigned int visible[MAX_GPUS] = {0};
    int numVisible = 10;
    numVisible = parse_cuda_visible_devices(str, visible, MAX_GPUS);
    assert(numVisible == 0);
    }
}

void test_visible_devices()
{
    dcgmHandle_t handle;
    CHECK_DCGM(dcgmInit());
    CHECK_DCGM(dcgmConnect("127.0.0.1", &handle));

    int success = 0;

    {
    jobenv_t je;
    je.visible_devices = "0,1,2,3";
    unsigned int visible[MAX_GPUS] = {0};
    int numVisible = 0;
    visible_devices(handle, &je, visible, &numVisible);
    ASSERT_CLEAN(numVisible == 4, shutdown);
    ASSERT_CLEAN(visible[0] == 0, shutdown);
    ASSERT_CLEAN(visible[1] == 1, shutdown);
    ASSERT_CLEAN(visible[2] == 2, shutdown);
    ASSERT_CLEAN(visible[3] == 3, shutdown);
    }

    // Removed this test because we detect the number of GPUs 
    // in the system and report on all of them.
    // {
    // jobenv_t je;
    // je.visible_devices = "0";
    // unsigned int visible[MAX_GPUS] = {0};
    // int numVisible = 0;
    // visible_devices(handle, &je, visible, &numVisible);
    // ASSERT_CLEAN(numVisible == 1, shutdown);
    // ASSERT_CLEAN(visible[0] == 0, shutdown);
    // }

    // {
    // jobenv_t je;
    // je.visible_devices = "3";
    // unsigned int visible[MAX_GPUS] = {0};
    // int numVisible = 0;
    // visible_devices(handle, &je, visible, &numVisible);
    // ASSERT_CLEAN(numVisible == 1, shutdown);
    // ASSERT_CLEAN(visible[0] == 3, shutdown);
    // }

    {
    jobenv_t je;
    je.visible_devices = "";
    unsigned int visible[MAX_GPUS] = {0};
    int numVisible = 0;
    visible_devices(handle, &je, visible, &numVisible);
    ASSERT_CLEAN(numVisible == 4, shutdown);
    ASSERT_CLEAN(visible[0] == 0, shutdown);
    ASSERT_CLEAN(visible[1] == 1, shutdown);
    ASSERT_CLEAN(visible[2] == 2, shutdown);
    ASSERT_CLEAN(visible[3] == 3, shutdown);
    }

    success = 1;

shutdown:
    dcgmDisconnect(handle);
    dcgmShutdown();

    if (!success) exit(1);
}

void test_parse_args()
{
    {
        cmdargs_t args;
        char *argv[] = {"jr"};
        int    argc = sizeof(argv) / sizeof(argv[0]);
        args.child_argc = 100;
        parse_args(argc, argv, &args);
        assert(args.show_help == 0);
        assert(args.child_argc == 0);
        assert(args.child_argv[0] == NULL);
    }

    {
        cmdargs_t args;
        char *argv[] = {"jr", "-x"};
        int    argc = sizeof(argv) / sizeof(argv[0]);
        parse_args(argc, argv, &args);
        assert(args.show_help == 0);
    }

    {
        cmdargs_t args;
        char *argv[] = {"jr", "-h"};
        int    argc = sizeof(argv) / sizeof(argv[0]);
        parse_args(argc, argv, &args);
        assert(args.show_help == 1);
    }

    {
        cmdargs_t args;
        char *argv[] = {"jr", "--help"};
        int    argc = sizeof(argv) / sizeof(argv[0]);
        parse_args(argc, argv, &args);
        assert(args.show_help == 1);
    }

    {
        cmdargs_t args;
        char *argv[] = {"jr", "--version"};
        int    argc = sizeof(argv) / sizeof(argv[0]);
        parse_args(argc, argv, &args);
        assert(args.show_version == 1);
    }

    {
        cmdargs_t args;
        char *argv[] = {"jr", "--"};
        int    argc = sizeof(argv) / sizeof(argv[0]);
        args.child_argc = 100;
        parse_args(argc, argv, &args);
        assert(args.show_help == 0);
        assert(args.child_argc == 0);
        assert(args.child_argv[0] == NULL);
    }

    {
        cmdargs_t args;
        char *argv[] = {"jr", "--", "ls"};
        int    argc = sizeof(argv) / sizeof(argv[0]);
        args.child_argc = 100;
        parse_args(argc, argv, &args);
        assert(args.child_argc == 1);
        assert(!strcmp("ls", args.child_argv[0]));
        assert(args.child_argv[1] == NULL);
    }

    {
        cmdargs_t args;
        char *argv[] = {"jr", "ls"};
        int    argc = sizeof(argv) / sizeof(argv[0]);
        args.child_argc = 100;
        parse_args(argc, argv, &args);
        assert(args.child_argc == 1);
        assert(!strcmp("ls", args.child_argv[0]));
        assert(args.child_argv[1] == NULL);
    }

    {
        cmdargs_t args;
        char *argv[] = {"jr", "ls", "1", "2", "3", "4", "--", "6"};
        int    argc = sizeof(argv) / sizeof(argv[0]);
        args.child_argc = 100;
        parse_args(argc, argv, &args);
        assert(args.child_argc == 7);
        assert(!strcmp("ls", args.child_argv[0]));
        assert(args.child_argv[7] == NULL);
    }

    {
        cmdargs_t args;
        char *argv[] = {"jr", "ls", "1", "2", "3", "4", "--", "6"};
        int    argc = sizeof(argv) / sizeof(argv[0]);
        args.child_argc = 100;
        parse_args(argc, argv, &args);
        assert(args.child_argc == 7);
        assert(!strcmp("ls", args.child_argv[0]));
        assert(!strcmp("6", args.child_argv[6]));
        assert(args.child_argv[7] == NULL);
    }

    {
        cmdargs_t args;
        char *argv[] = {"jr", "-o", "dir", "cmd", "3"};
        int    argc = sizeof(argv) / sizeof(argv[0]);
        args.child_argc = 100;
        parse_args(argc, argv, &args);
        assert(args.child_argc == 2);
        assert(!strcmp("dir", args.outdir));
        assert(!strcmp("cmd", args.child_argv[0]));
        assert(!strcmp("3", args.child_argv[1]));
        assert(args.child_argv[2] == NULL);
    }

    {
        cmdargs_t args;
        char *argv[] = {"jr", "-o", "dir", "--", "cmd", "3"};
        int    argc = sizeof(argv) / sizeof(argv[0]);
        args.child_argc = 100;
        parse_args(argc, argv, &args);
        assert(args.child_argc == 2);
        assert(!strcmp("dir", args.outdir));
        assert(!strcmp("cmd", args.child_argv[0]));
        assert(!strcmp("3", args.child_argv[1]));
        assert(args.child_argv[2] == NULL);
    }
}

void test_create_output_location()
{
    {
        FILE *fp, *metafp;
        jobenv_t jobenv = (jobenv_t){
            .slurm_step = "12",
            .slurm_rank = "2",
            .slurm_localid = "0",
            .slurm_jobid = "4567",
            .slurm_jobname = "test",
            .rank0 = 0,
            .local0 = 1,
            .visible_devices="1,2,3,4",
            .with_slurm = 1
        };
        cmdargs_t args = (cmdargs_t){
            .show_help = 0,
            .outdir = NULL,
        };
        create_output_location(&fp, &metafp, &jobenv, &args);
        assert(!fclose(fp));
        struct stat st;
        assert(!stat("report_4567/step_12/proc_2.csv", &st));
        printf("Created report_4567/step_12/proc_2.csv.\n");
        assert(!stat("report_4567/step_12/proc_2.meta.txt", &st));
        printf("Created report_4567/step_12/proc_2.meta.txt.\n");
    }

    {
        FILE *fp, *metafp;
        jobenv_t jobenv = (jobenv_t){
            .slurm_step = "12",
            .slurm_rank = "2",
            .slurm_localid = "0",
            .slurm_jobid = "4567",
            .slurm_jobname = "test",
            .rank0 = 0,
            .local0 = 1,
            .visible_devices="1,2,3,4",
            .with_slurm = 1
        };
        cmdargs_t args = (cmdargs_t){
            .show_help = 0,
            .outdir = "alt_report",
        };
        create_output_location(&fp, &metafp, &jobenv, &args);
        assert(!fclose(fp));
        struct stat st;
        assert(!stat("report_4567/step_12/proc_2.csv", &st));
        printf("Created report_4567/step_12/proc_2.csv.\n");
        assert(!stat("report_4567/step_12/proc_2.meta.txt", &st));
        printf("Created report_4567/step_12/proc_2.meta.txt.\n");
    }

    {
        FILE *fp;
        jobenv_t jobenv = (jobenv_t){
            .slurm_step = "12",
            .slurm_rank = "2",
            .slurm_localid = "0",
            .slurm_jobid = "4567",
            .slurm_jobname = "test",
            .rank0 = 0,
            .local0 = 1,
            .visible_devices="1,2,3,4",
            .with_slurm = 1
        };
        cmdargs_t args = (cmdargs_t){
            .show_help = 0,
            .outdir = "alt_report",
        };
        create_output_location(&fp, NULL, &jobenv, &args);
        assert(!fclose(fp));
        struct stat st;
        assert(!stat("report_4567/step_12/proc_2.csv", &st));
        printf("Created report_4567/step_12/proc_2.csv.\n");
        assert(!stat("report_4567/step_12/proc_2.meta.txt", &st));
        printf("Created report_4567/step_12/proc_2.meta.txt.\n");
    }
}

void test_write_meta()
{
    {
        jobenv_t jobenv = (jobenv_t){
            .slurm_step = "12",
            .slurm_rank = "2",
            .slurm_localid = "0",
            .slurm_jobid = "4567",
            .slurm_jobname = "test",
            .rank0 = 0,
            .local0 = 1,
            .visible_devices="1,2,3,4",
            .slurm_cluster = "test-cluster",
            .slurm_nnodes = "4",
            .slurm_ntasks = "128",
            .slurm_ngpus = "4",
            .nnodes = 4,
            .ntasks = 128,
            .ngpus = 4,
            .with_slurm = 1,
        };
        cmdargs_t args = (cmdargs_t){
            .child_argv = (char *[]){"exe", "arg1", "arg2", "arg3", "multi string arg"},
            .child_argc = 5
        };
        write_meta(stdout, &args, &jobenv);
    }
}

void test_write_records()
{
//   {
//   record_t records = (record_t){
//       .ts;
//       .gpuId;
//       .values = {{.fieldId = 
//   };
//   write_records(stdout, 1, records, 0);
//   }
}

void test_help()
{
    help();
    exit(0);
}

void test_version()
{
    version();
    exit(0);
}

void run_tests()
{
    test_job_environment();
    test_parse_cuda_visible_devices();
    test_visible_devices();
    test_parse_args();
    test_create_output_location();
    test_write_records();
    test_write_meta();
    test_help();
    test_version();

    printf("\n\nTESTS OK\n");
}
