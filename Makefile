CC=gcc

all:
	$(CC) gssr-record.c gssr-record-tests.c -ldcgm -o gssr-record 

clean:
	rm gssr-record

clean-tests:
	-rm -rf report_4567 alt_report
	-rm ./test-ubuntu.toml
	-rm -rf test-report-*

install-uv:
	curl --proto '=https' --tlsv1.2 -LsSf https://github.com/astral-sh/uv/releases/download/0.9.24/uv-installer.sh | sh
