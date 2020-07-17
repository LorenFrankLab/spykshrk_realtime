cimport cRSTPython as RST

cdef extern from 'RSTFunctions.h':

	ctypedef struct query_buf:
		unsigned int len
		unsigned int cur
		int *buf

	ctypedef struct eval_buf:
		float x1,x2,x3,x4
		unsigned int len
		unsigned int cur
		float *weight
		float *pos

	void init_gaussian_table(
			float mean, float stddev, 
			float min, float max, float interval)

	float gaussian_func(float x)


	float euc_dist(
			float x1, float x2, float x3, float x4, 
			float y1, float y2, float y3, float y4)


	bint Intersects(RST.RSTREE R,
			RST.typrect RSTrect,
			RST.typrect queryrect,
			RST.typrect unused)


	bint IsContained(RST.RSTREE R,
			RST.typrect RSTrect,
			RST.typrect queryrect,
			RST.typrect unused)


	void CountQuery(RST.RSTREE R,
			RST.typrect rectangle,
			RST.refinfo infoptr,
			void *buf,
			bint *modify,
			bint *finish)

	void PosQuery(RST.RSTREE R,
			RST.typrect rectangle,
			RST.refinfo infoptr,
			void *buf,
			bint *modify,
			bint *finish)

	void EvalQuery(RST.RSTREE R,
			RST.typrect rectangle,
			RST.refinfo infoptr,
			void *buf,
			bint *modify,
			bint *finish)
