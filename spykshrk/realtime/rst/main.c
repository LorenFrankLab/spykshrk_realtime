#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>

#include "RStarTree.h"

#define NUM_ELEM 1000000
#define NUM_QUERY 10000

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
	//printf("intersects (%.2f %.2f %.2f %.2f) (%.2f %.2f %.2f %.2f) query (%.2f %.2f %.2f %.2f) (%.2f %.2f %.2f %.2f), %d\n",RSTrect[0].l,RSTrect[1].l,RSTrect[2].l,RSTrect[3].l,RSTrect[0].h,RSTrect[1].h,RSTrect[2].h,RSTrect[3].h,queryrect[0].l,queryrect[1].l,queryrect[2].l,queryrect[3].l,queryrect[0].h,queryrect[1].h,queryrect[2].h,queryrect[3].h,inter);
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
	//printf("contained (%.2f %.2f %.2f %.2f) (%.2f %.2f %.2f %.2f) query (%.2f %.2f %.2f %.2f) (%.2f %.2f %.2f %.2f), %d\n",RSTrect[0].l,RSTrect[1].l,RSTrect[2].l,RSTrect[3].l,RSTrect[0].h,RSTrect[1].h,RSTrect[2].h,RSTrect[3].h,queryrect[0].l,queryrect[1].l,queryrect[2].l,queryrect[3].l,queryrect[0].h,queryrect[1].h,queryrect[2].h,queryrect[3].h,iscont);
	return iscont;
}

void CountQuery(RSTREE R,
		typrect rectangle,
		refinfo infoptr,
		void *buf,
		boolean *modify,
		boolean *finish)

{
	//printf("counting: (%.2f %.2f %.2f %.2f) (%.2f %.2f %.2f %.2f) content %d\n",rectangle[0].l,rectangle[1].l,rectangle[2].l,rectangle[3].l,rectangle[0].h,rectangle[1].h,rectangle[2].h,rectangle[3].h,infoptr->contents);
	int *count = buf;
	(*count) ++;
	*modify = FALSE;
	*finish = FALSE;
}


int main(int argc, char ** argv) {
	RSTREE rst;
	boolean insert_success;
	RemoveRST("test_tree");
	boolean create_success = CreateRST("test_tree",4096, FALSE);
	srand ( (unsigned int)time ( NULL ) );
	if(create_success) {
		printf("Created test_tree.\n");
	}

	NoRSTree(&rst);
	boolean open_success = OpenRST(&rst, "test_tree");
	if(open_success) {
		printf("Opened test_tree.\n");
	}

	struct timeval start, end;
	double insert_time;
	double insert_avg;
	gettimeofday(&start, NULL);
	int i = 0;
	for(i = 0; i < NUM_ELEM; i++) {
		typrect rect;
		typinfo info;
		int x1 = rand()%2048;
		int x2 = rand()%2048;
		int x3 = rand()%2048;
		int x4 = rand()%2048;
		/*
		   x1 = 500+i;
		   x2 = 500+i;
		   x3 = 500+i;
		   x4 = 500+i;
		   */

		rect[0].l = x1;
		rect[0].h = x1;
		rect[1].l = x2;
		rect[1].h = x2;
		rect[2].l = x3;
		rect[2].h = x3;
		rect[3].l = x4;
		rect[3].h = x4;

		info.contents = i;

		InsertRecord(rst,rect,&info,&insert_success);
	}
	gettimeofday(&end, NULL);
	insert_time = ((double)(end.tv_sec - start.tv_sec)*1000 + (double)(end.tv_usec - start.tv_usec)/1000.0);
	if(insert_success) {
		printf("Inserted into test_tree. Average Insertion time %lfms for %d elem\n",insert_time/NUM_ELEM,NUM_ELEM);
	}

	char name[80];
	int numbofdim;
	int sizedirentry;
	int sizedataentry;
	int sizeinfo;
	int maxdirfanout;
	int maxdatafanout;
	int pagesize;
	int numbofdirpages;
	int numbofdatapages;
	int pagesperlevel[100];
	int numbofrecords;
	int height;
	boolean  unique;

	InquireRSTDesc(rst,name,&numbofdim,&sizedirentry,&sizedataentry,&sizeinfo,&maxdirfanout,&maxdatafanout,&pagesize,&numbofdirpages,&numbofdatapages,pagesperlevel,&numbofrecords,&height,&unique);

	printf("numbofdim %d numrecs %d\n",numbofdim,numbofrecords);

	typrect unused;

	double query_time;
	gettimeofday(&start, NULL);
	double countavg = 0;
	for(i = 0; i < NUM_QUERY; i++) {
		typrect rect;
		rect[0].l = rand()%1048;
		rect[0].h = rect[0].l+1000;
		rect[1].l = rand()%1048;
		rect[1].h = rect[1].l+1000;
		rect[2].l = rand()%1048;
		rect[2].h = rect[2].l+1000;
		rect[3].l = rand()%1048;
		rect[3].h = rect[3].l+1000;


		int count = 0;
		RegionQuery(rst,rect,unused,Intersects,Intersects,CountQuery, &count);
		countavg += (double)(count-countavg)/(i+1.0);
	}
	gettimeofday(&end, NULL);
	query_time = ((double)(end.tv_sec - start.tv_sec)*1000 + (double)(end.tv_usec - start.tv_usec)/1000.0);
	printf("Number of Queries: %d Average Elements Recieved: %lf Average Query Time: %lfms\n",NUM_QUERY,countavg,query_time/NUM_QUERY);
}
