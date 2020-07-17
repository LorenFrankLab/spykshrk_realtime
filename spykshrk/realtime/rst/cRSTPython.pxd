
cdef extern from "RStarTree.h":

	ctypedef float typatomkey
	ctypedef struct RSTREE:
		pass
	ctypedef struct typinterval:
		typatomkey l
		typatomkey h

	ctypedef struct typinfo:
		int contents
	
	ctypedef typinfo *refinfo

	ctypedef typinterval typrect[4]

	ctypedef bint (*DirQueryProc) (RSTREE, typrect, typrect,typrect)
	ctypedef bint (*DataQueryProc) (RSTREE, typrect, typrect, typrect)
	ctypedef void (*QueryManageProc) (RSTREE, typrect, refinfo, void*, bint*, bint*)
	ctypedef bint (*JoinManageProc) (RSTREE, RSTREE, typrect, typrect, refinfo, refinfo, void*, void*, bint*)


	bint RemoveRST(char *name)
	bint CreateRST(char *name, int pagesize, bint unique)
	void NoRSTree(RSTREE *rst)
	bint OpenRST(RSTREE *rst, char *name)
	bint CloseRST(RSTREE *rst)
	bint InsertRecord(RSTREE rst, typrect rectangle, typinfo *info, bint *inserted)

	bint  InquireRSTDesc(RSTREE rst,
			char *name,
			int *numbofdim,
			int *sizedirentry,
			int *sizedataentry,
			int *sizeinfo,
			int *maxdirfanout,
			int *maxdatafanout,
			int *pagesize,
			int *numbofdirpages,
			int *numbofdatapages,
			int pagesperlevel[],
			int *numbofrecords,
			int *height,
			bint *unique)

	bint RegionQuery(RSTREE rst, 
			typrect queryrect1, 
			typrect queryrect2, 
			DirQueryProc DirQuery, 
			DataQueryProc DatQuery, 
			QueryManageProc Manage,
			void *pointer)


#cdef extern from "RSTFunction.h":
#	bint Insersects(RSTREE R, typrect RSTrect, typrect queryrect, typrect unused)
#	bint IsContained(RSTREE R, typrect RSTrect, typrect queryrect, typrect unused)
#	void CountQuery(RSTREE R, typrect rectangle, refinfo infoptr, void *buf, boolean *modify, bint *finish);


