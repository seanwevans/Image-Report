#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define C1 (errno == ERANGE && (offset == LONG_MAX || offset == LONG_MIN))
#define C2 (errno != 0 && offset == 0)
#define C3                                                                     \
  (errno != 0 || *end != '\0' || end == argv[2] || offset < 0 ||               \
   offset >= filesize)

#define D1 (errno == ERANGE && (num_chars == LONG_MAX || num_chars == LONG_MIN))
#define D2 (errno != 0 && num_chars == 0)
#define D3                                                                     \
  (errno != 0 || *end != '\0' || end == argv[3] || num_chars <= 0 ||           \
   num_chars + offset > filesize)

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

const char *error_message[] = {
    "",
    "Usage:  extract_average_hash  input_file.xml  offset  num_chars",
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

FILE *file = NULL;
char *data = NULL;

void stop(ErrorCode error_code) {
  if (data)
    free(data);

  if (file)
    fclose(file);

  if (error_code > 1)
    fprintf(stderr, "Error: ");

  fprintf(stderr, "%s\n", error_message[error_code]);

  exit(error_code);
}

int main(int argc, char *argv[]) {
  size_t filesize = 0;
  long offset = 0;
  long num_chars = 0;
  char *end = NULL;

  errno = 0;
  if (argc < 4)
    stop(USAGE_MSG);

  file = fopen(argv[1], "rb");
  if (!file)
    stop(FILE_OPEN_ERROR);

  if (fseek(file, 0, SEEK_END) != 0)
    stop(FILE_SIZE_ERROR);

  filesize = ftell(file);
  if (filesize <= 0)
    stop(FILE_VALIDATION_ERROR);

  if (fseek(file, 0, SEEK_SET) != 0)
    stop(FILE_REWIND_ERROR);

  errno = 0;
  offset = strtol(argv[2], &end, 10);
  if (C1 || C2 || C3)
    stop(OFFSET_ERROR);

  errno = 0;
  num_chars = strtol(argv[3], &end, 10);
  if (D1 || D2 || D3)
    stop(NUM_CHARS_ERROR);

  data = calloc(num_chars + 1, 1);
  if (!data)
    stop(MEMORY_ALLOC_ERROR);

  if (fseek(file, offset, SEEK_SET) != 0)
    stop(OFFSET_SEEK_ERROR);

  if (fread(data, 1, num_chars, file) != num_chars)
    stop(DATA_READ_ERROR);
  data[num_chars] = '\0';

  if (puts(data) == EOF)
    stop(DATA_WRITE_ERROR);

  if (fclose(file) != 0)
    stop(FILE_CLOSE_ERROR);

  free(data);

  return EXIT_SUCCESS;
}
