#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <stdarg.h>
#include "edgeTest.h"
using namespace cv;
using namespace std;
#ifndef FALSE
#define FALSE 0
#endif /* !FALSE */

#ifndef TRUE
#define TRUE 1
#endif /* !TRUE */

/*----------------------------------------------------------------------------*/
/* fatal error, print a message to standard error and exit
*/
//static void error(char * msg)
//{
//	fprintf(stderr, "error: %s\n", msg);
//	exit(EXIT_FAILURE);
//}
//
///*----------------------------------------------------------------------------*/
///* memory allocation, print an error and exit if fail
//*/
//static void * xmalloc(size_t size)
//{
//	void * p;
//	if (size == 0) error("xmalloc input: zero size");
//	p = malloc(size);
//	if (p == NULL) error("out of memory");
//	return p;
//}

/* open file, print an error and exit if fail
*/
static bool x_sort(const vector<float>& n1, const vector<float>& n2) {
	return n1[1] < n2[1];
}
static FILE * xfopen(const char * path, const char * mode)
{
	FILE * f = fopen(path, mode);
	if (f == NULL)
	{
		fprintf(stderr, "error: unable to open file '%s'\n", path);
		exit(EXIT_FAILURE);
	}
	return f;
}

/* close file, print an error and exit if fail
*/
static int xfclose(FILE * f)
{
	if (fclose(f) == EOF) error("unable to close file");
	return 0;
}

/* skip white characters and comments in a PGM file
*/
static void skip_whites_and_comments(FILE * f)
{
	int c;
	do
	{
		while (isspace(c = getc(f))); /* skip spaces */
		if (c == '#') /* skip comments */
			while (c != '\n' && c != '\r' && c != EOF)
				c = getc(f);
	} while (c == '#' || isspace(c));
	if (c != EOF && ungetc(c, f) == EOF)
		error("unable to 'ungetc' while reading PGM file.");
}

/* read a number in ASCII from a PGM file
*/
static int get_num(FILE * f)
{
	int num, c;

	while (isspace(c = getc(f)));
	if (!isdigit(c)) error("corrupted PGM or PPM file.");
	num = c - '0';
	while (isdigit(c = getc(f))) num = 10 * num + c - '0';
	if (c != EOF && ungetc(c, f) == EOF)
		error("unable to 'ungetc' while reading PGM file.");

	return num;
}

/* read a PGM image file
*/
double * read_pgm_image(char * name, int * X, int * Y)
{
	FILE * f;
	int i, n, depth, bin = FALSE;
	double * image;

	/* open file */
	f = xfopen(name, "rb"); /* open to read as a binary file (b option). otherwise,
							in some systems, it may behave differently */

	/* read header */
	if (getc(f) != 'P') error("not a PGM file!");
	if ((n = getc(f)) == '2') bin = FALSE;
	else if (n == '5') bin = TRUE;
	else error("not a PGM file!");
	skip_whites_and_comments(f);
	*X = get_num(f);               /* X size */
	skip_whites_and_comments(f);
	*Y = get_num(f);               /* Y size */
	skip_whites_and_comments(f);
	depth = get_num(f);            /* pixel depth */
	if (depth < 0) error("pixel depth < 0, unrecognized PGM file");
	if (bin && depth > 255) error("pixel depth > 255, unrecognized PGM file");
	/* white before data */
	if (!isspace(getc(f))) error("corrupted PGM file.");

	/* get memory */
	image = (double *)xmalloc(*X * *Y * sizeof(double));

	/* read data */
	for (i = 0; i<(*X * *Y); i++)
		image[i] = (double)(bin ? getc(f) : get_num(f));

	/* close file */
	xfclose(f);

	/* return image */
	return image;
}

/*----------------------------------------------------------------------------*/
/* read a 2D ASC format file
*/
double * read_asc_file(char * name, int * X, int * Y)
{
	FILE * f;
	int i, n, Z, C;
	double val;
	double * image;

	/* open file */
	f = xfopen(name, "rb"); /* open to read as a binary file (b option). otherwise,
							in some systems, it may behave differently */

	/* read header */
	n = fscanf(f, "%d%*c%d%*c%d%*c%d", X, Y, &Z, &C);
	if (n != 4 || *X <= 0 || *Y <= 0 || Z <= 0 || C <= 0) error("invalid ASC file");

	/* only gray level images are handled */
	if (Z != 1 || C != 1) error("only single channel ASC files are handled");

	/* get memory */
	image = (double *)xmalloc(*X * *Y * Z * C * sizeof(double));

	/* read data */
	for (i = 0; i<(*X * *Y * Z * C); i++)
	{
		n = fscanf(f, "%lf", &val);
		if (n != 1) error("invalid ASC file");
		image[i] = val;
	}

	/* close file */
	xfclose(f);

	return image;
}

/*----------------------------------------------------------------------------*/
/* read an image from a file in ASC or PGM formats
*/
double * read_image(char * name, int * X, int * Y)
{
	int n = (int)strlen(name);
	char * ext = name + n - 4;

	if (n >= 4 && (strcmp(ext, ".asc") == 0 || strcmp(ext, ".ASC") == 0))
		return read_asc_file(name, X, Y);

	return read_pgm_image(name, X, Y);
}

/*----------------------------------------------------------------------------*/
/* write curves into a PDF file. the output is PDF version 1.4 as described in
"PDF Reference, third edition" by Adobe Systems Incorporated, 2001
*/
bool write_curves_pdf(double * x, double * y, int * curve_limits, int M,
					  const char * filename, int X, int Y, double width, int& n1, int& n2)
{
	FILE * pdf;
	long start1, start2, start3, start4, start5, startxref, stream_len;
	int i, j, k;

	/* check input */
	if (filename == NULL) error("invalid filename in write_curves_pdf");
	if (M > 0 && (x == NULL || y == NULL || curve_limits == NULL))
		error("invalid curves data in write_curves_pdf");
	if (X <= 0 || Y <= 0) error("invalid image size in write_curves_pdf");

	/* open file */
	pdf = xfopen(filename, "wb"); /* open to write as a binary file (b option).
								  otherwise, in some systems,
								  it may behave differently */

	/* PDF header */
	fprintf(pdf, "%%PDF-1.4\n");
	/* The following PDF comment contains characters with ASCII codes greater
	than 128. This helps to classify the file as containing 8-bit binary data.
	See "PDF Reference" p.63. */
	fprintf(pdf, "%%%c%c%c%c\n", 0xe2, 0xe3, 0xcf, 0xd3);

	/* Catalog, Pages and Page objects */
	start1 = ftell(pdf);
	fprintf(pdf, "1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\n");
	fprintf(pdf, "endobj\n");
	start2 = ftell(pdf);
	fprintf(pdf, "2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1 ");
	fprintf(pdf, "/Resources <<>> /MediaBox [0 0 %d %d]>>\nendobj\n", X, Y);
	start3 = ftell(pdf);
	fprintf(pdf, "3 0 obj\n");
	fprintf(pdf, "<</Type /Page /Parent 2 0 R /Contents 4 0 R>>\n");
	fprintf(pdf, "endobj\n");

	/* Contents object - graphic contents */
	start4 = ftell(pdf);
	fprintf(pdf, "4 0 obj\n<</Length 5 0 R>>\n"); /* indirect length in obj 5 */
	fprintf(pdf, "stream\n");
	stream_len = ftell(pdf);
	fprintf(pdf, "%.4f w\n", width); /* set line width */
	// Classify characters and lines and verify the validity of the input image
	int points_number = 0;
	int T_index = -1;
	vector<Rect> vec_box;
	vector<int> vec_box_index;
	bool valid_ = false;
	int i_1, i_2, i_3, i_4;
	for (k = 0; k < M; k++)
	{
		if(k >= 1)
			points_number = curve_limits[k] - curve_limits[k - 1];
		else
			points_number = 0;
		if(points_number >= 50) {
			vector<Point2f> contour_points;
			i = curve_limits[k - 1];
			for (j = i ; j < curve_limits[k]; j++)
				contour_points.push_back(Point2f(x[j], y[j]));
			Rect box = boundingRect(contour_points);
			//cout << box << endl;
			vec_box.push_back(box);
			vec_box_index.push_back(i);
		}
	}
	for(int number = 0; number != vec_box.size(); number++)
	{
		Rect box_i = vec_box[number];
		float area_i = box_i.area();
		float width_i = box_i.width;
		float height_i = box_i.height;
		Point2f upper_left_i = Point2f(box_i.x, box_i.y);
		vector<vector<float> > tmp_index;
		vector<float> tmp;
		tmp.push_back(vec_box_index[number]);tmp.push_back(upper_left_i.x);
		tmp_index.push_back(tmp);
		for(int number_next = number + 1; number_next != vec_box.size(); number_next++ )
		{
			Rect box_j = vec_box[number_next];
			float area_j = box_j.area();
			float width_j = box_j.width;
			float height_j = box_j.height;
			Point2f upper_left_j = Point2f(box_j.x, box_j.y);
			float area_ratio = area_i / area_j;
			bool area_cri = area_ratio >= 0.5 && area_ratio <= 2 ? true : false;
			float width_ratio = width_i / width_j;
			bool width_cri = width_ratio >= 0.8 && width_ratio <= 1.6 ? true : false;
			float height_ratio = height_i / height_j;
			bool height_cri = height_ratio >= 0.8 && height_ratio <= 1.6 ? true : false;
			float y_diff = abs(upper_left_i.y - upper_left_j.y);
			bool y_cri = y_diff <= 35 ? true : false;
			if(area_cri && width_cri && height_cri && y_cri) {
				vector<float> tmp;
				tmp.push_back(vec_box_index[number_next]);
				tmp.push_back(upper_left_j.x);
				tmp_index.push_back(tmp);
			}
			if(tmp_index.size() == 4)
				break;
		}
		if(tmp_index.size() == 4) {
			sort(tmp_index.begin(), tmp_index.end(), x_sort);
			i_1 = tmp_index[0][0];
			i_2 = tmp_index[1][0];
			i_3 = tmp_index[2][0];
			i_4 = tmp_index[3][0];
//            cout << tmp_index[0][1] << endl;
//            cout << tmp_index[1][1] << endl;
//            cout << tmp_index[2][1] << endl;
//            cout << tmp_index[3][1] << endl;
			T_index = static_cast<int>(tmp_index[1][0]);
			valid_ = true;
			break;
		}
	}
	//
	int max_number_points = 1;
	int max_number_points_index = -1;
	if(valid_ == true) {
		for (k = 0; k < M; k++) /* write curves */
		{
			if (k >= 1)
				points_number = curve_limits[k] - curve_limits[k - 1];
			else
				points_number = 0;
			if (points_number >= max_number_points) {
				max_number_points = points_number;
				max_number_points_index = curve_limits[k - 1];
			}
		}
		if(max_number_points_index == i_1 || max_number_points_index == i_2 || max_number_points_index == i_3 || max_number_points_index == i_4)
			valid_ = false;
//		if(max_number_points <= 450)  //new -> only select some clear stop signs
//			valid_ = false;           //
	}
	n2 = max_number_points_index;
	if(valid_ == true) {
		for (k = 0; k < M; k++) /* write curves */
		{
			/* an offset of 0.5,0.5 is added to point coordinates so that the
            drawing has the right positioning when superposed on the image
            drawn to the same size. in that case, pixels are drawn as squares
            of size 1,1 and the coordinate of the detected edge points are
            relative to the center of those squares. thus the 0.5, 0.5 offset.
            */

			/* initate chain */
			if (k >= 1)
				points_number = curve_limits[k] - curve_limits[k - 1];
			else
				points_number = 0;
			if (points_number >= 20) {
				//cout << points_number << endl;
				//cout << curve_limits[k - 1] << " " << curve_limits[k] << " " << curve_limits[k + 1] << endl;
				i = curve_limits[k - 1];
//				if (i == T_index || i == max_number_points_index) {
				if (i == max_number_points_index) {
					fprintf(pdf, "%.4f %.4f m\n", x[i] + 0.5, Y - y[i] - 0.5); /* first point */

					/* add remaining points of the curve */
					for (j = i + 1; j < curve_limits[k]; j++)
						fprintf(pdf, "%.4f %.4f l\n", x[j] + 0.5, Y - y[j] - 0.5);

					/* if the curve is closed, market as such */
					j = curve_limits[k] - 1;
					if (x[i] == x[j] && y[i] == y[j]) fprintf(pdf, "h\n");

					/* end curve - stroke! */
					fprintf(pdf, "S\n");
				}
			}
		}
	}
	stream_len = ftell(pdf) - stream_len; /* store stream length */
	fprintf(pdf, "\r\nendstream\n"); /* EOL must be CRLF before endstream */
	fprintf(pdf, "endobj\n");

	/* Contents' stream length object - the use of this indirect object
    for the stream length allows to generate the PDF file in a single
    pass, specifying the stream��s length only when its contents have
    been generated. See "PDF Reference" p.40. */
	start5 = ftell(pdf);
	fprintf(pdf, "5 0 obj\n%ld\nendobj\n", stream_len);

	/* PDF Cross-reference table */
	startxref = ftell(pdf);
	fprintf(pdf, "xref\n");
	fprintf(pdf, "0 6\n");
	fprintf(pdf, "0000000000 65535 f\r\n"); /* EOL must be CRLF in xref table */
	fprintf(pdf, "%010ld 00000 n\r\n", start1);
	fprintf(pdf, "%010ld 00000 n\r\n", start2);
	fprintf(pdf, "%010ld 00000 n\r\n", start3);
	fprintf(pdf, "%010ld 00000 n\r\n", start4);
	fprintf(pdf, "%010ld 00000 n\r\n", start5);

	/* PDF trailer */
	fprintf(pdf, "trailer <</Size 6 /Root 1 0 R>>\n");
	fprintf(pdf, "startxref\n");
	fprintf(pdf, "%ld\n", startxref);
	fprintf(pdf, "%%%%EOF\n");

	/* close file */
	xfclose(pdf);
	n1 = T_index;
	return valid_;
}

void gradient_separate()
{

}
/*----------------------------------------------------------------------------*/
/* write curves into a TXT file
*/
void write_curves_txt(double * x, double * y, double * theta, int * curve_limits, int M,
					  Mat& src, int& T_index, int& inner_polygon_index)
{
	//filename -> store all points
	//filename1 -> store points(inner_polygon)
	//filename2 -> store points(outer_polygon)
	//filename3 -> store points(inner_T)
	int i, k;

	/* check input */

	/* write curves */
	int points_number = 0;
	for (k = 0; k<M; k++) /* write curves */
	{
		if(k != 0)
			points_number = curve_limits[k] - curve_limits[k - 1];
		else
			points_number = 0;
		if(points_number >= 10) {
			if(curve_limits[k - 1] == T_index || curve_limits[k - 1] == inner_polygon_index) {
				for (i = curve_limits[k - 1]; i < curve_limits[k]; i++) {
					src.at<Vec3b>(static_cast<int>(y[i]), static_cast<int>(x[i]))[0] = 0;
					src.at<Vec3b>(static_cast<int>(y[i]), static_cast<int>(x[i]))[1] = 0;
					src.at<Vec3b>(static_cast<int>(y[i]), static_cast<int>(x[i]))[2] = 0;
					//fprintf(txt, "%g %g Gradient direction:%g\n", x[i], y[i], theta[i]);
				}
			}
		}
	}
}
//costomized write_curve_txt -> for project
int write_curves_txt_costomized(double * x, double * y, double * theta, int * curve_limits, int M,
								 const char * filename1, const char * filename2, int& T_index, int& inner_polygon_index)
{
	//filename -> store all points
	//filename1 -> store points(inner_polygon)
	//filename2 -> store points(outer_polygon)
	//filename3 -> store points(inner_T)
	FILE * txt;
	FILE * txt1;
	int i, k;

	int work = 0;
	bool T_tmp = false;
	bool inner_polygon_tmp = false;
	/* write curves */
	vector<int> Points_;
	for (k = 0; k<M; k++) /* write curves */
	{
		// cout << "k" << k << endl;
		if(k == 0){
			continue;
		}
		// cout << "k=" << k << endl;
		// cout << "curve_limits[k-1]" << curve_limits[k-1] << endl;
		// continue;
		if(curve_limits[k - 1] == inner_polygon_index && inner_polygon_tmp == false){
			// cout << " if " << endl;
			/* close file */
			inner_polygon_tmp = true;
			txt = xfopen(filename1, "wb"); // -> inner polygon
			//fprintf(txt, "Number of points:%d\n", max_); /* start of chain */
			for (i = curve_limits[k - 1]; i < curve_limits[k]; i++) {
				fprintf(txt, "%g %g\n", x[i], y[i]);
			}
			xfclose(txt);
			// cout << " end if " << endl;
			if(work == 1) {
				work += 1;
				break;
			}
			else
				work += 1;
		}
		if(curve_limits[k - 1] == T_index && T_tmp == false){
			T_tmp = true;
			txt1 = xfopen(filename2, "wb"); // -> inner polygon
			//fprintf(txt, "Number of points:%d\n", max_); /* start of chain */
			for (i = curve_limits[k - 1]; i < curve_limits[k]; i++) {
				fprintf(txt1, "%g %g\n", x[i], y[i]);
			}
			xfclose(txt1);
			if(work == 1) {
				work += 1;
				break;
			}
			else
				work += 1;
		}
	}
	return work;
}



/*----------------------------------------------------------------------------*/
/* write curves into a SVG file
*/
void write_curves_svg(double * x, double * y, int * curve_limits, int M,
					  char * filename, int X, int Y, double width)
{
	FILE * svg;
	int i, k;

	/* check input */
	if (filename == NULL) error("invalid filename in write_curves_svg");
	if (M > 0 && (x == NULL || y == NULL || curve_limits == NULL))
		error("invalid curves data in write_curves_svg");
	if (X <= 0 || Y <= 0) error("invalid image size in write_curves_svg");

	/* open file */
	svg = xfopen(filename, "wb"); /* open to write as a binary file (b option).
								  otherwise, in some systems,
								  it may behave differently */

	/* write SVG header */
	fprintf(svg, "<?xml version=\"1.0\" standalone=\"no\"?>\n");
	fprintf(svg, "<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n");
	fprintf(svg, " \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n");
	fprintf(svg, "<svg width=\"%dpx\" height=\"%dpx\" ", X, Y);
	fprintf(svg, "version=\"1.1\"\n xmlns=\"http://www.w3.org/2000/svg\" ");
	fprintf(svg, "xmlns:xlink=\"http://www.w3.org/1999/xlink\">\n");

	/* write curves */
	for (k = 0; k<M; k++) /* write curves */
	{
		fprintf(svg, "<polyline stroke-width=\"%g\" ", width);
		fprintf(svg, "fill=\"none\" stroke=\"black\" points=\"");
		for (i = curve_limits[k]; i<curve_limits[k + 1]; i++)
			fprintf(svg, "%g,%g ", x[i], y[i]);
		fprintf(svg, "\"/>\n"); /* end of chain */
	}

	/* close SVG file */
	fprintf(svg, "</svg>\n");
	xfclose(svg);
}
