#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef enum {
  USAGE_MSG = 1,
  FILE_OPEN_ERROR,
  FILE_SIZE_ERROR,
  FILE_VALIDATION_ERROR,
  FILE_REWIND_ERROR,
  OFFSET_ERROR,
  NUM_CHARS_ERROR,
  MEMORY_ALLOC_ERROR,
  OFFSET_SEEK_ERROR,
  DATA_READ_ERROR,
  DATA_WRITE_ERROR,
  FILE_CLOSE_ERROR
} ErrorCode;

static const char *error_message[] = {
    "",
    "Usage: extract_average_hash input_file.xml offset num_chars",
    "Failed to open input file",
    "Failed to retrieve file size",
    "Failed to validate file size",
    "Failed to rewind file",
    "Invalid offset given",
    "Invalid number of characters given",
    "Failed to allocate memory",
    "Failed to seek to offset",
    "Failed to read input data",
    "Failed to write output",
    "Failed to close input file"};

static FILE *file = NULL;
static char *data = NULL;

static void cleanup(void) {
  if (data) {
    free(data);
    data = NULL;
  }

  if (file) {
    (void)fclose(file);
    file = NULL;
  }
}

static void stop(ErrorCode error_code) {
  cleanup();

  if (error_code > 1) {
    fprintf(stderr, "Error: ");
  }

  fprintf(stderr, "%s\n", error_message[error_code]);
  exit(error_code);
}

int main(int argc, char *argv[]) {
  long filesize = 0;
  long offset = 0;
  long num_chars = 0;
  char *end = NULL;
  size_t n = 0;

  if (argc != 4) {
    stop(USAGE_MSG);
  }

  file = fopen(argv[1], "rb");
  if (!file) {
    stop(FILE_OPEN_ERROR);
  }

  if (fseek(file, 0, SEEK_END) != 0) {
    stop(FILE_SIZE_ERROR);
  }

  filesize = ftell(file);
  if (filesize < 0) {
    stop(FILE_SIZE_ERROR);
  }
  if (filesize == 0) {
    stop(FILE_VALIDATION_ERROR);
  }

  if (fseek(file, 0, SEEK_SET) != 0) {
    stop(FILE_REWIND_ERROR);
  }

  errno = 0;
  offset = strtol(argv[2], &end, 10);
  if (errno != 0 || end == argv[2] || *end != '\0' || offset < 0 ||
      offset >= filesize) {
    stop(OFFSET_ERROR);
  }

  errno = 0;
  num_chars = strtol(argv[3], &end, 10);
  if (errno != 0 || end == argv[3] || *end != '\0' || num_chars <= 0 ||
      num_chars > filesize - offset) {
    stop(NUM_CHARS_ERROR);
  }

  n = (size_t)num_chars;
  if ((long)n != num_chars || n == SIZE_MAX) {
    stop(NUM_CHARS_ERROR);
  }

  data = calloc(n + 1, 1);
  if (!data) {
    stop(MEMORY_ALLOC_ERROR);
  }

  if (fseek(file, offset, SEEK_SET) != 0) {
    stop(OFFSET_SEEK_ERROR);
  }

  if (fread(data, 1, n, file) != n) {
    stop(DATA_READ_ERROR);
  }

  data[n] = '\0';

  if (fwrite(data, 1, n, stdout) != n) {
    stop(DATA_WRITE_ERROR);
  }
  if (fputc('\n', stdout) == EOF) {
    stop(DATA_WRITE_ERROR);
  }

  if (fclose(file) != 0) {
    file = NULL;
    stop(FILE_CLOSE_ERROR);
  }
  file = NULL;

  free(data);
  data = NULL;

  return EXIT_SUCCESS;
}