CC=gcc

all:
	$(CC) gssr-record.c gssr-record-tests.c -ldcgm -o gssr-record 

clean:
	rm gssr-record

clean-tests:
	-rm -rf report_4567 alt_report
	-rm ./test-ubuntu.toml
	-rm -rf test-report-??
