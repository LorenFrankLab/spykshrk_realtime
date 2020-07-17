#include "RStarTree.h"
#include <math.h>

// Growth rate of buffer
#define BUF_GROWTH_RATE  1.5

// Square root of two PI (used in gaussian calculation)
#define SQRT_TWO_PI  2.50662827463


// Fixed-Point Arithmetic Definitions
#define FIXED_POINT 10
#define MUL(x,y) (((unsigned int) x >> FIXED_POINT/2) * ( (unsigned int) y >> (FIXED_POINT/2 + FIXED_POINT % 2)))
#define ADD(x,y) (x+y)
#define SUB(x,y) (x-y)
#define FIXED_CONV pow(2,FIXED_POINT);
#define TOFLOAT(x) (( (float) (x) ) /  FIXED_CONV)
#define TOFIXED(x) ((int) ( x * FIXED_CONV ))
#define FIXED_SQRT_CONV TOFIXED(FIXED_CONV/sqrt(FIXED_CONV))
#define SQRT(x) ((int)sqrt(x)) * FIXED_SQRT_CONV

#define EUC_DIST(a1,b1,c1,d1,a2,b2,c2,d2) sqrt((a1-a2)*(a1-a2)+(b1-b2)*(b1-b2)+(c1-c2)*(c1-c2)+(d1-d2)*(d1-d2))
/* #define EUC_DIST(a1,b1,c1,d1,a2,b2,c2,d2) (SQRT( MUL(SUB(a1,a2),SUB(a1,a2)) + \
		MUL(SUB(b1,b2),SUB(b1,b2)) + \
		MUL(SUB(c1,c2),SUB(c1,c2)) + \
		MUL(SUB(d1,d2),SUB(d1,d2))))
*/


/**
 * Structure to store parameters and lookup table for gaussian kernel
 * for kernel density estimation */
typedef struct {
	float min;
	float max;
	float interval;
	int size;
	float* table;
} gaussian_struct;

// a single global kernel lookup table
// untested whether global C variable scope goes across
// Cython class instantiation
extern gaussian_struct gaussian_lookup;

/**
 * Union to convert between floats and ints
 */
typedef union {
	float f;
	int i;
} float2int;

/**
 * Buffer to store results of a query
 */
typedef struct {
	unsigned int len;
	unsigned int cur;
	int *buf;
} query_buf;

/**
 * Buffer to store results of a full evaluation
 * of the kernel density algorithm.
 */
typedef struct {
	int x1,x2,x3,x4;  	// evaluating point in tet-space
	unsigned int len;	// full size of buffer
	unsigned int cur;	// cursor at end of used buffer
	float *weight;		// buffer for kernel weight contribution
	float *pos;			// spatial location

} eval_buf;

/**
 * Function to initialize hardcoded gaussian table
 *
 * @param mean		gaussian mean
 * @param stddev	gaussian std dev
 * @param min		minimum value of lookup table
 * @param max		maxmimum value of lookup table
 * @param interval	interval between bins of lookup table
 */
void init_gaussian_table(float mean, float stddev, float min, float max, float interval);

/**
 * Function to evaluate a single point on the gaussian using lookup table.
 * @param x		value to evaluate
 */
float gaussian_func(float x);

/**
 * Function to compute the euclidean distance between two 4-D points.
 * @param x1
 * @param x2
 * @param x3
 * @param x4
 * @param y1
 * @param y2
 * @param y3
 * @param y4
 */
float euc_dist(float x1, float x2, float x3, float x4, float y1, float y2, float y3, float y4); 

/**
 * RSTREE function for computing intersections
 * within the tree.
 *
 * @param R			R*-Tree data structure
 * @param RSTrect	hypercube being evaluated
 * @param queryrect	hypercube from initial query
 * @param unsued
 */
boolean Intersects(RSTREE R,
		typrect RSTrect,
		typrect queryrect,
		typrect unused);

/**
 * RSTREE function for evaluating when querying hypercube
 * contains a particular area of the tree.
 *
 * @param R			R*-Tree data structure
 * @param RSTrect	hypercube being evaluated
 * @param queryrect	hypercube from initial query
 * @param unsued
 */

boolean IsContained(RSTREE R,
		typrect RSTrect,
		typrect queryrect,
		typrect unused);

/**
 * RSTREE function to query and accumulate the number of
 * elements a query returns.  This function is called
 * each time the query hits a valid element.
 *
 * @param R				R*-Tree data structure pointer
 * @param rectangle		Rectangle of query point
 * @param infoptr		content of query point
 * @param buf			pointer to counter value to increment
 * @param modify		return value of whether underlying RSTree was modified
 * @param finish		return value to set if the query should end prematurely
 */
void CountQuery(RSTREE R,
		typrect rectangle,
		refinfo infoptr,
		void *buf,
		boolean *modify,
		boolean *finish);

/**
 * RSTREE function to query and accumulate the number of
 * elements a query returns.  This function is called
 * each time the query hits a valid element.
 *
 * @param R				R*-Tree data structure pointer
 * @param rectangle		Rectangle of query point
 * @param infoptr		content of query point
 * @param buf			pointer to counter value to increment
 * @param modify		return value of whether underlying RSTree was modified
 * @param finish		return value to set if the query should end prematurely
 */

void EvalQuery(RSTREE R,
		typrect rectangle,
		refinfo infoptr,
		void *buf,
		boolean *modify,
		boolean *finish);

/**
 * RSTREE function to query and append the positions of
 * each element to a buffer (query_buf).
 *
 * @param R				R*-Tree data structure pointer
 * @param rectangle		Rectangle of query point
 * @param infoptr		content of query point
 * @param buf			pointer to query_buf structure
 * @param modify		return value of whether underlying RSTree was modified
 * @param finish		return value to set if the query should end prematurely
 */
 
void PosQuery(RSTREE R,
		typrect rectangle,
		refinfo infoptr,
		void *buf,
		boolean *modify,
		boolean *finish);

