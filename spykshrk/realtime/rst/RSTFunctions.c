#include "RStarTree.h"
#include "RSTFunctions.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

gaussian_struct gaussian_lookup;

void init_gaussian_table(float mean, float stddev, float min, float max, float interval) {
	float x;
	int i = 0;

	gaussian_lookup.min = min;
	gaussian_lookup.max = max;
	gaussian_lookup.interval = interval;
	gaussian_lookup.size = (floorl((max-min)/interval)+2);
	gaussian_lookup.table = malloc(gaussian_lookup.size*sizeof(float));
	for( x = min; x <= max; x += interval) {
		gaussian_lookup.table[i] = 
			1.0/(stddev*SQRT_TWO_PI)*(exp(-1.0/2.0*((x-mean)*(x-mean))/(stddev*stddev)));
		i++;
	}

	if( i+1 < gaussian_lookup.size ) {
		x += interval;
		gaussian_lookup.table[i] = 
			1.0/(stddev*SQRT_TWO_PI)*(exp(-1.0/2.0*((x-mean)*(x-mean))/(stddev*stddev)));
		i++;
	}
	
}

float gaussian_func(float x) {
	if ( x < gaussian_lookup.min || x > gaussian_lookup.max ) {
		printf("ERROR: gaussian_lookup for x=%f out of range! Return 0 and not valid!\n", x);
		return 0;
	}

	float x_conv = (x - gaussian_lookup.min) / gaussian_lookup.interval;
	int x_ind = floorl(x_conv);
	return gaussian_lookup.table[x_ind] + 
		(gaussian_lookup.table[x_ind + 1] - gaussian_lookup.table[x_ind]) * 
		(x_conv - floor(x_conv));
}

float euc_dist(float x1, float x2, float x3, float x4, float y1, float y2, float y3, float y4) {
	return sqrt((x1 - y1)*(x1 - y1) + 
		(x2 - y2)*(x2 - y2) + 
		(x3 - y3)*(x3 - y3) + 
		(x4 - y4)*(x4 - y4));
}

boolean Intersects(RSTREE R,
                   typrect RSTrect,
                   typrect queryrect,
                   typrect unused)
{
  int maxdim= NumbOfDim -1;
  boolean inter;
  int d;
  

  d= -1;
  do {
    d++;
    inter= RSTrect[d].l <= queryrect[d].h &&
           RSTrect[d].h >= queryrect[d].l;
  } while (inter && d != maxdim);
  return inter;
}

boolean IsContained(RSTREE R,
		typrect RSTrect,
		typrect queryrect,
		typrect unused)
{
	int maxdim= NumbOfDim -1;
	boolean iscont;
	int d;



	d= -1;
	do {
		d++;
		iscont= RSTrect[d].l >= queryrect[d].l &&
			RSTrect[d].h <= queryrect[d].h;
	} while (iscont && d != maxdim);
	return iscont;
}

void CountQuery(RSTREE R,
		typrect rectangle,
		refinfo infoptr,
		void *buf,
		boolean *modify,
		boolean *finish)
{
	int *count = buf;
	(*count) ++;
	*modify = FALSE;
	*finish = FALSE;
}

void PosQuery(RSTREE R,
		typrect rectangle,
		refinfo infoptr,
		void *buf,
		boolean *modify,
		boolean *finish)
{
	query_buf* _buf = buf;
	
	// Increase buffer size if limit reached
	if (_buf->cur >= _buf->len) {
		_buf->buf = realloc(_buf->buf, (unsigned int)(_buf->len*sizeof(float)*BUF_GROWTH_RATE));
		_buf->len = _buf->len * sizeof(int) * BUF_GROWTH_RATE;
	}

	_buf->buf[_buf->cur] = (int) infoptr->contents;
	_buf->cur ++;
	
	*modify = FALSE;
	*finish = FALSE;
}

void EvalQuery(RSTREE R,
		typrect rectangle,
		refinfo infoptr,
		void *buf,
		boolean *modify,
		boolean *finish)
{
	eval_buf *_buf = buf;

	// Increase buffer size if limit reached
	if (_buf->cur >= _buf->len) {
		printf("Realloc-ing to %d %d %f %d \n",_buf->len,sizeof(int),sizeof(int)*BUF_GROWTH_RATE,(unsigned int)(_buf->len*(sizeof(int)*BUF_GROWTH_RATE))); fflush(stdin);
		_buf->weight = realloc(_buf->weight, (unsigned int)(_buf->len*(sizeof(float)*BUF_GROWTH_RATE)));
		_buf->pos = realloc(_buf->pos, (unsigned int)(_buf->len*(sizeof(int)*BUF_GROWTH_RATE)));
		_buf->len = (unsigned int)(_buf->len  * BUF_GROWTH_RATE);
		if (_buf->weight == NULL)
			printf("Failed to realloc _buf->weight...\n"); fflush(stdin);
		if(_buf->pos == NULL)
			printf("Failed to realloc _buf->pos...\n"); fflush(stdin);
		printf("realloc successful!\n");fflush(stdin);

	}

	//printf("%d\n",_buf->cur);fflush(stdin);
	float x1 = rectangle[0].l;
	float x2 = rectangle[1].l;
	float x3 = rectangle[2].l;
	float x4 = rectangle[3].l;

	float2int pos_content;

	_buf->weight[_buf->cur] = gaussian_func(euc_dist(_buf->x1, _buf->x2, _buf->x3, _buf->x4, x1, x2, x3, x4));
	pos_content.i = infoptr->contents;
	_buf->pos[_buf->cur] = pos_content.f;
	_buf->cur ++;

	*modify = FALSE;
	*finish = FALSE;

}
