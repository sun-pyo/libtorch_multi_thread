#ifndef TEST_H
#define TEST_H

#include <pthread.h>
#include "thpool.h"

extern threadpool thpool;

extern pthread_cond_t *cond_t;
extern pthread_mutex_t *mutex_t;
extern int *cond_i;

#endif
