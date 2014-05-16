#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <fstream>
#include <iomanip>
//#include <iostream>
//#include <math.h>
using namespace std;
using namespace cv; 

Mat imgSrc;
int factor;

/*************************************************************************************************
 * Funciones auxiliares                                                                          *
 *************************************************************************************************/
//Array de colores, para usarlos en al pintar
Scalar colores[8] = { Scalar(  0,   0, 204),   //Rojo
					  Scalar(  0, 200, 255),   //Amarillo			  
					  Scalar(  0, 255,   0),   //Verde
					  Scalar(255,  51, 102),   //Azul
					  Scalar(238, 130, 238),   //Violeta
					  Scalar(255, 255, 255),   //Blanco
					  Scalar(128, 128, 128),   //Gris
					  Scalar(  0,   0,   0) }; //Negro

Vec3b keyColor(0,0,0);
#define ROJO     0
#define AMARILLO 1
#define VERDE    2
#define AZUL     3
#define VIOLETA  4
#define BLANCO   5
#define GRIS     6
#define NEGRO    7

/*************************************************************************************************
 * Funciones Auxiliares                                                                          *
 *************************************************************************************************/
/*
	función que visualiza una imagen
*/
int pintaImagen(Mat & img, String & titulo){	
	//Si existe imagen, se muestra, en caso contrario se informa del error
	if(img.data!=NULL){
		namedWindow(titulo, 1);
		imshow(titulo, img);
		waitKey();
		destroyWindow(titulo);
		return 0;
	}else{
		cout << endl << "Error. No se pudo abrir la imagen" << endl;
		return -1;
	}
}
/*
   función que lee los ficheros de un directorio
   Entrada: carpeta. Ruta de la carpeta con las imágenes
            f.       Vector para almacenar los nombres de los ficheros
   Salida:  f (sobrescrito). Vector con los nombres de los ficheros de la carpeta
*/
void getDir(string carpeta, vector<string> &f){
	FILE* pipe =  NULL;
	string pCmd = "dir /B " + carpeta;
	char buf[256];
	pipe = _popen(pCmd.c_str(),"rt");
	if(pipe == NULL){
		cout<<"Error cargando el directorio "<<carpeta << endl;
		exit(1);
	}

	while (!feof(pipe))
		if(fgets(buf,256,pipe) != NULL){
			f.push_back(string(buf));
			f.back().resize(f.back().size()-1);
		}

	_pclose(pipe);
}
/*
   función que carga las imágenes desde una carpeta   
   Entrada: carpeta. Ruta de la carpeta con las imágenes
            vecImg.  Vector para almacenar las imágenes
   Salida:  vecImg (sobrescrito). Vector con las imágenes de la carpeta
*/
void leerFicheros(string carpeta,vector<Mat> &vecImg){
	vector<string> files;
	getDir(carpeta, files);
	string fichero;
	Mat imagen;

	for(int i=0; i<files.size(); i++){
		fichero = carpeta + "/" + files[i];
		imagen=imread(fichero,CV_LOAD_IMAGE_ANYCOLOR);
		cout << "Cargando imagen: " << fichero << endl;
		if(imagen.data==NULL){
			cout << "Error cargando imagen: " << fichero << endl;
		}else{
			if(imagen.channels()==1)
				cvtColor(imagen,imagen,CV_GRAY2BGR);
			imagen.convertTo(imagen,CV_8UC3);
			vecImg.push_back(imagen);
		}
	}
}

/* 
   Devuelve cuantas veces se puede dividir filasCols entre 2 
   Entrada: filasCols. valor a dividir
   Salida: maxNiveles (return). Número de veces a dividir
*/
int calcularNiveles(int filasCols){
	int maxNiveles=0;
	while(filasCols%2==0){
		maxNiveles++;
		filasCols/=2;
	}
	return maxNiveles;
}

/* 
	En un rango ente topeMin y max, partiendo de max busca el valor que más veces se puede dividir entre 2
	que sea menor o igual que maxNiveles
	Entrada: max.         Valor máximo del rango
	         topeMin.     Valor mínimo del rango
			 maxNiveles.  Tope de veces que se quiere dividir entre 2
			 posMax.      Para almacenar el mejor valor encontrado
			 maxOut.      Para almacenar las veces que se puede dividir
	Salida:  posMax (Sobrescrito). El mejor valor encontrado
	         maxOut (Sobrescrito). Las veces que se puede dividir entre 2 el mejor valor encontrado
*/
void calcularPotDosCercana(int max, int topeMin, int maxNiveles, int & posMax, int & maxOut){
	int i=max;
	int niveles;
	bool seguir=true;
	posMax=max;
	maxOut=0;
	while(i>topeMin && seguir){
		niveles=calcularNiveles(i);
		if(niveles>maxOut){
			maxOut=niveles;
			posMax=i;
		}
		if(maxOut>=maxNiveles){
			seguir=false;
			maxOut=maxNiveles;
		}
		i--;
	}
}
/* 
	Calcula el píxel que le corresponde a un punto p según la homografía h dada
	Entrada: p. Punto (x,y)
	         h. Matriz 3x3 que representa una homografía
	Salida:  dst (return). Punto (x',y') que le corresponde al punto p
	         
*/
Point warpPoint(Point p, Mat h){
	Point dst;
	double div;

	div   =  h.at<double>(2,0)*p.x + h.at<double>(2,1)*p.y + h.at<double>(2,2);
	dst.x = (h.at<double>(0,0)*p.x + h.at<double>(0,1)*p.y + h.at<double>(0,2))/div;
	dst.y = (h.at<double>(1,0)*p.x + h.at<double>(1,1)*p.y + h.at<double>(1,2))/div;

	return dst;
}

/*************************************************************************************************
 * Funciones para eliminar los bordes negros de la imagen                                        *
 *************************************************************************************************/
/* 
   Toma una imagen y una columna y determina si en la columna existe un píxel con 
   color distinto al indicado por keyColor. En tal caso devuelve false.
   Entrada: img. Imagen
            x.   Nº de columna a escanear
   Salida:  Booleano (Return). Devuelve true si toda la columna es del color keyColor
                               y false si hay algún píxel de distinto color
*/
bool checkColColor(Mat & img, int x){
    for(int y=0; y<img.rows; y++)    {
		if(img.at<Vec3b>(y,x) != keyColor)
            return false;
    }
    return true;
}
/* 
   Toma una imagen, una columna y un umbral y contabiliza el número de píxeles con
   color distinto al indicado por keyColor. Si se supera el umbral devuelve false.
   Entrada: img. Imagen
            x.   Nº de columna a escanear
   Salida:  Booleano (Return). Devuelve true si no se supera el umbral
                               y false en caso contrario
*/
bool checkColColorAgresive(Mat & img, int x, int umbral){
	int contNoBlack=0;
	bool ret=true;
    for(int y=0; y<img.rows && contNoBlack<umbral; y++)
		if(img.at<Vec3b>(y,x) != keyColor) contNoBlack++;

	if(contNoBlack>=umbral)	ret=false;

    return ret;
}
/* 
   Toma una imagen, una columna y un umbral y contabiliza el número de píxeles con
   color distinto al indicado por keyColor. Si se supera el umbral devuelve false.
   Entrada: img. Imagen
            x.   Nº de columna a escanear
   Salida:  Booleano (Return). Devuelve true si no se supera el umbral
                               y false en caso contrario
*/
bool checkRowColorAgresive(Mat & img, int y, int umbral){
	int contNoBlack=0;
	bool ret=true;
    for(int x=0; x<img.cols && contNoBlack<umbral; x++)
		if(img.at<Vec3b>(y,x) != keyColor) contNoBlack++;

    if(contNoBlack>=umbral)	ret=false;

    return ret;
}
/* 
   Toma una imagen localiza en una fila el mayor número de píxeles con color distinto
   al indicado por keyColor.
   Entrada: img. Imagen
   Salida:  Entero (Return). Devuelve el número de píxeles distintos de keyColor de la
							fila que más píxeles distintos de keyColor tiene
*/
int contMaxNoBlackRow(Mat & img){
	int contNoBlack,maxContNoBlack=0;
	for(int i=0; i<img.rows; i++){
		contNoBlack=0;
		for(int x=0; x<img.cols; x++)
			if(img.at<Vec3b>(i,x) != keyColor) contNoBlack++;	

		if(contNoBlack>maxContNoBlack) maxContNoBlack=contNoBlack;
	}

    return maxContNoBlack;
}

/* 
   Toma una imagen localiza en una columna el mayor número de píxeles  con color distinto
   al indicado por keyColor.
   Entrada: img. Imagen
   Salida:  Entero (Return). Devuelve el número de píxeles distintos de keyColor de la
							columna que más píxeles distintos de keyColor tiene
*/
int contMaxNoBlackCol(Mat & img){
	int contNoBlack,maxContNoBlack=0;
	for(int i=0; i<img.cols; i++){
		contNoBlack=0;
		for(int y=0; y<img.rows; y++)
			if(img.at<Vec3b>(y,i) != keyColor) contNoBlack++;	

		if(contNoBlack>maxContNoBlack) maxContNoBlack=contNoBlack;
	}

    return maxContNoBlack;
}

/* 
   Toma una imagen y si tiene laterales del color keyColor trata de eliminarlos
   Entrada: img. Imagen
   Salida:  imgProc (Return). Imagen sin los bordes
*/
Mat cortarLaterales(Mat & img){
	//Para controlar la posición de las cuatro esquinas
	int posXi = 0, posXf = img.cols-1;
	bool cambio;	
	//Buscamos por cada lado, los bordes a eliminar		
	do{
		cambio = false;		
		if(checkColColor(img, posXi)){ posXi++;	cambio = true; }//Columna izquierda		
		if(checkColColor(img, posXf)){ posXf--;	cambio = true; }//Columna derecha
	}while(cambio);
	return img(Rect(posXi, 0, posXf-posXi+1, img.rows));
}
/* 
   Toma una imagen y si tiene bordes del color keyColor trata de eliminarlos
   Entrada: img. Imagen
   Salida:  imgProc (Return). Imagen sin los bordes
*/
Mat cortarBordes(Mat & img){
	//Para controlar la posición de las cuatro esquinas
	int posXi = 0, posXf = img.cols-1;
	int posYi = 0, posYf = img.rows-1;
	bool cambio;
	double umbral=0.95;
	
	int maxCol = contMaxNoBlackCol(img)*umbral;
	int maxRow = contMaxNoBlackRow(img)*umbral;
	
	do{
		cambio = false;		
		if(checkRowColorAgresive(img, posYi,maxRow)){ posYi++; cambio = true; }//Fila superior		
		if(checkRowColorAgresive(img, posYf,maxRow)){ posYf--; cambio = true; }//Fila inferior
	}while(cambio);

	maxCol = contMaxNoBlackCol(img(Rect(0, posYi, posXf+1, posYf-posYi+1)));
	do{
		cambio = false;		
		if(checkColColorAgresive(img, posXi,maxCol)){ posXi++;	cambio = true; }//Columna izquierda		
		if(checkColColorAgresive(img, posXf,maxCol)){ posXf--;	cambio = true; }//Columna derecha		
	}while(cambio);
	return img(Rect(posXi, posYi, posXf-posXi+1, posYf-posYi+1));
}


/* 
   Toma una imagen y calcula los keyPoint según el algoritmo SIFT
   Entrada: imgIn.     Imagen a la que extraer los puntos
   Salida:  keypoints (return). Vector de keyPoint con los puntos clave
*/
vector<KeyPoint> aplicarSift(Mat & imgIn){
	Mat img;                     //Imagen temporal
	vector<KeyPoint> keypoints;  //Vector para almacenar los keyPoint
	//Parámetros para SIFT
	int    nFeatures     = 0;    //Nº características a retener
	int    nOctaveLayers = 3;    //Capas por nivel
	double contrThres    = 0.05; //Umbral contraste
	double edgeThres     = 10.0; //Umbral de bordes
	double sigma         = 1.6;  //Sigma para gaussiana
	cvtColor(imgIn, img, CV_RGB2GRAY); 

	//Definimos objeto SIFT y extraemos los puntos
	SIFT sift(nFeatures, nOctaveLayers, contrThres, edgeThres, sigma); 
	sift(img, noArray(), keypoints, noArray(), false);	

	//Calculamos el nivel de la pirámide para representarlo después
	for(unsigned int i=0; i<keypoints.size(); i++){
		keypoints[i].octave = keypoints[i].octave & 0xFF;
		if(keypoints[i].octave==255)
			keypoints[i].octave = 0;
		else
			keypoints[i].octave++;
	}

	return keypoints;
}
/* 
   Busca correspondencias entre dos imágenes (dados sus descriptores SIFT) usando el critero de Lowe
   Entrada: descriptors1. Matriz de descriptores SIFT de la primera imagen
			descriptors2. Matriz de descriptores SIFT de la segunda imagen
			matches.      Vector de DMatch para guardar las coincidencias
   Salida:  matches (sobrescrito). Vector de DMatch con las coincidencias de ambas imágenes
*/
void correspondenciaKNN(Mat & descriptors1, Mat & descriptors2, vector<DMatch> & matches){
	vector<vector<DMatch>> preMatches; //Correspondencias devueltas por knnMatch
	//Parámetros para el matcher
	int k = 2; //Nº de correspondencias por descriptor devueltas
	float umbral = 0.4;
	FlannBasedMatcher matcher;	

	//Emparejamos
	matcher.knnMatch(descriptors1, descriptors2, preMatches, k);
	matches.reserve(preMatches.size());	
	//knnMatch devuelve los dos puntos mas cercanos al descriptor x, por tanto ahora hay que
	//usar el criterio de Lowe (||f1-f2||/||f1-f2'||<umbral) para ver si nos quedamos con el match o no
	//preMatches[i][0].distance es igual a ||f1-f2|| así que si hacemos preMatches[i][0].distance/preMatches[i][1].distance
	//estamos diciendo que:
	//si valor es 0.1, hay mucha distancia entre [0] y [1], tomamos el [0] que es muy bueno
	//si valor es 0.9, hay poca  distancia entre [0] y [1], no tomamos nada, es muy ambiguo
	for(unsigned int i=0; i<preMatches.size(); i++){
		if(preMatches[i].size()>1){			
			if((preMatches[i][0].distance/preMatches[i][1].distance)<umbral)
				matches.push_back(preMatches[i][0]);
		}
	}	
}



/*************************************************************************************************
 * Funciones para realizar la mezcla piramidal                                                   *
 *************************************************************************************************/
/*
   Calcula la pirámide de una imagen y devuelve todos sus niveles
   Entrada: im_in.   Imagen de entrada
			ims_out. Vector para almacenar la imagen de cada nivel de la pirámide
			niveles. Número de niveles a calcular de la pirámide
   Salida:  ims_out (sobrescrito). Pirámide
*/
void calcularPiramide(Mat & im_in, vector<Mat> & ims_out, int niveles){	
	Mat tmp=im_in;
	ims_out.push_back(im_in);
	for(int l=0; l<niveles; l++){
		pyrDown(tmp,tmp);		
		ims_out.push_back(tmp);
	}
}
/*
   Calcula las laplacianas de las imágenes de entrada y las devuelve
   Entrada: im_in.   Vector con las imágenes a las que hay que calcular las laplacianas
			ims_out. Vector para almacenar las laplacianas
   Salida:  ims_out (sobrescrito). Laplacianas de cada imagen
*/
void calcularLaplacianas(vector<Mat> & ims_in, vector<Mat> & ims_out){
	Mat exp;
	ims_out.resize(ims_in.size());
	ims_out[ims_in.size()-1]=ims_in.back();
	for(int i=ims_in.size()-2; i>=0; i--){		
		pyrUp(ims_in[i+1],exp);
		ims_out[i]=ims_in[i]-exp;
	}	
}
/*
   Combina (por la mitad) las laplacianas que hay en ims1 con las que hay en ims2
   y las devuelve. De forma que la parte izquierda pertenece a ims1 y la parte derecha
   a ims2
   Entrada: ims1.    Vector con las laplacianas de la izquierda
			ims2.    Vector con las laplacianas de la derecha
			ims_out. Vector para almacenar las laplacianas combinadas
   Salida:  ims_out (sobrescrito). Laplacianas combinadas
*/
void combinarLaplacianas(vector<Mat> & ims1, vector<Mat> & ims2, vector<Mat> & ims_out){
	int mitad;
	Mat tmp,roi1,roi2;
	ims_out.clear();
	for(int i=0; i<ims1.size(); i++){
		mitad=ims1[i].cols/2;
		ims1[i].copyTo(tmp);		
		roi1=tmp(Rect(mitad,0,mitad,ims1[i].rows));
		roi2=ims2[i](Rect(mitad,0,mitad,ims1[i].rows));
		roi2.copyTo(roi1);
		ims_out.push_back(tmp);
	}
}

/*
   Calcula la mezcla piramidal con las laplacianas. El último nivel permanece intacto. Para los
   siguientes niveles, si n es el contenido del nivel, se ejecuta el siguiente bucle:
   n=n + (n+1)expandido
   refiriéndose el primer operador de suma a suma de matrices y el segundo al nivel
   que está por encima en la pirámide.
   Entrada: im_in.   Vector con las imágenes a mezclar
			ims_out. Para almacenar la imagen devuelta
   Salida:  im_out (sobrescrito). Imagen mezclada
*/
void mezclarLaplacianas(vector<Mat> & ims_in, Mat & im_out){
	Mat exp;
	vector <Mat> ims_out;
	for(int i=0; i<ims_in.size(); i++)
		ims_out.push_back(ims_in[i]);

	for(int i=ims_in.size()-2; i>=0; i--){
		pyrUp(ims_out[i+1],exp);
		ims_out[i]=ims_out[i]+exp;
	}
	im_out=ims_out.front();
}


/*
   Realiza la mezcla piramidad de dos imágenes según el algoritmo de Burt-Adelson
   Entrada: im1.     Imagen que quedará a la izquierda después de mezclar
            im2.     Imagen que quedará a la derecha después de mezclar
			ims_out. Para almacenar la imagen devuelta
			niveles. Número de niveles a aplicar en la construcción de la pirámide
   Salida:  ims_out (sobrescrito). Imagen mezclada
*/
void mezclaPiramidal(Mat &im1, Mat &im2, Mat &im_out, int niveles){
	vector<Mat> p1,p2,l1,l2,lc;
	Mat imf1,imf2;
	int width;

	if(im1.rows!= im2.rows){ cout << "Las imagenes tienen numero de filas distinto" << endl; exit(1);}
	if(im1.cols<im2.cols) width = im1.cols;
	else                  width = im2.cols;

	im1.convertTo(imf1,CV_64FC3);
	im2.convertTo(imf2,CV_64FC3);

	calcularPiramide(imf1,p1,niveles);
	calcularPiramide(imf2,p2,niveles);	

	calcularLaplacianas(p1,l1);
	calcularLaplacianas(p2,l2);
	combinarLaplacianas(l1,l2,lc);

	mezclarLaplacianas(lc,im_out);
	im_out.convertTo(im_out,CV_8UC3);
}
/*************************************************************************************************
 * Conversión a cilíndricas                                                                      *
 *************************************************************************************************/
/*
   Transforma una imagen a cilíndricas con Inverse Warping y según el factor de escala indicado.
   Entrada: imgIn.  Imagen a transformar
            imgOut. Para almacenar la imagen devuelta
			f.      Factor de escala a aplicar a la transformación
   Salida:  imgOut (sobrescrito). Imagen transformada
*/
void imgACilindricas(vector<Mat> & imgIn, vector<Mat> & imgOut, double f){
	Mat imgTmp;
	double xCil, yCil, xImg, yImg;
	
	imgOut.clear();
	imgOut.resize(imgIn.size());
	for(int k=0; k<imgIn.size(); k++){
		if(imgIn[k].channels()!=3)
			cvtColor(imgIn[k], imgIn[k], CV_GRAY2BGR);

		//Creamos la imagen de destino de color negro
		imgTmp.create(imgIn[k].rows, imgIn[k].cols, imgIn[k].type());
		imgTmp = colores[NEGRO]; 

		//Comenzamos a buscar las correspondencias. Como en OpenCV el punto (0,0) está en
		//la esquina superior izquierda, hay que realizar una translación previa a la
		//transformación en sí, para tener el (0,0) en el centro.
		for(int i=0; i<imgIn[k].rows; i++){
			for(int j=0; j<imgIn[k].cols; j++){				
				//Desplazamos el punto (i,j)
				xCil =  j - (0.5*imgIn[k].cols)+1;
				yCil =  i - (0.5*imgIn[k].rows)+1;
				//Calculamos la coordenada x en la imagen original que le corresponde a la imagen transformada
				xImg = f*tan(xCil/f);
				//Calculamos la coordenada y en la imagen original que le corresponde a la imagen transformada
				yImg = yCil*sqrt(1+pow(tan(xCil/f),2));
				//Restauramos el desplazamiento
				xImg =  xImg + (0.5*imgIn[k].cols)-1;
				yImg =  yImg + (0.5*imgIn[k].rows)-1;
				//Si el punto (xImg, yImg) está dentro de los límites de la imagen original, tomamos su color
				if( yImg>=0.0 && yImg<imgIn[k].rows && xImg>=0.0 && xImg<imgIn[k].cols)
					imgTmp.at<Vec3b>(i,j) = imgIn[k].at<Vec3b>(yImg, xImg);
			}			
		}
		imgTmp.copyTo(imgOut[k]);
	}
}

/*************************************************************************************************
 * Construcción del panorama                                                                     *
 *************************************************************************************************/
/*
   Construye un panorama con mezcla de imágenes según el algoritmo de Burt-Adelson
   Las imágenes deben estar almacenadas en el orden de unión.
   Entrada: vecImg. Imágenes del panorama
   Salida:  canvas (return). Panorama construido
*/
Mat panorama(vector<Mat> & vecImg){
	vector< vector<KeyPoint> > keypoints;   //Vector para guardar los puntos destacados
	vector<Point2f> pointSrc, pointDst;     //Vectores para guardar los puntos que coinciden
	vector<DMatch>  matches;                //Vector para las correspondencias
	vector<Mat>     descriptors, homografias;
	SiftDescriptorExtractor extractor;      //Extractor de descriptores
	Mat imgMezclada, imgPers, H0;
	Mat roi1, roi2;
	Point p1,p2,p3;	
	int ancho, levels, maxLevels, regionComun;		
	int LARGO = (vecImg[0].cols*2)*vecImg.size();
	int ALTO  = vecImg[0].rows*2;
	int MITAD = (vecImg.size()-1)/2;	
	Mat canvas(ALTO, LARGO, CV_8UC3); //Canvas en el que se guardará el panorama

	//Calculamos el número máximo de niveles posibles a aplicar en las pirámides
	maxLevels = calcularNiveles(vecImg[0].rows);
	if(maxLevels>5) 
		maxLevels=5;

	homografias.resize(vecImg.size());	
	keypoints.resize(  vecImg.size());
	descriptors.resize(vecImg.size());

	//Calculamos los puntos y descriptores SIFT
	for(int i=0; i<vecImg.size(); i++){		
		cout << "Extrayendo descriptores SIFT de la imagen numero " << i << endl;
		//Extraemos los keyPoint
		keypoints[i] = aplicarSift(vecImg[i]);
		//Calculamos descriptores
		extractor.compute(vecImg[i], keypoints[i], descriptors[i]); 
	}

	//Calculamos las homografías de todas las imágenes
	//Prestablecemos la homografía que va entre la imagen central y el canvas
	homografias[MITAD] = (Mat_<double>(3, 3) <<   1, 0, (LARGO/2),
                                                  0, 1, (ALTO-vecImg[0].rows)/2,
                                                  0, 0, 1);
	
	//Calculamos las homografías de la parte derecha
	for(int i=MITAD+1; i<vecImg.size(); i++){ //int i=MITAD+1
		cout << "Calculando homografia de la imagen numero " << i << endl;
		//Inicializamos las variables
		pointSrc.clear(); pointDst.clear(); matches.clear();
		//Calculamos correspondencias por el método de knnMatch y criterio Lowe
		correspondenciaKNN(descriptors[i], descriptors[i-1], matches);
		//Guardamos los puntos		
		for(unsigned int j=0; j<matches.size(); j++){		
			pointSrc.push_back(keypoints[i][matches[j].queryIdx].pt);
			pointDst.push_back(keypoints[i-1][matches[j].trainIdx].pt);
		}
		//Calculamos la homografía
		homografias[i] = (findHomography(pointSrc, pointDst, CV_RANSAC, 1, noArray()));
		homografias[i] = homografias[i-1]*homografias[i];
	}
	//Calculamos las homografías de la parte izquierda
	for(int i=MITAD-1; i>=0; i--){ //int i=MITAD+1
		cout << "Calculando homografia de la imagen numero " << i << endl;
		//Inicializamos las variables
		pointSrc.clear(); pointDst.clear(); matches.clear();
		//Calculamos correspondencias por el método de knnMatch y criterio Lowe
		correspondenciaKNN(descriptors[i], descriptors[i+1], matches);
		//Guardamos los puntos		
		for(unsigned int j=0; j<matches.size(); j++){		
			pointSrc.push_back(keypoints[i][matches[j].queryIdx].pt);
			pointDst.push_back(keypoints[i+1][matches[j].trainIdx].pt);
		}
		//Calculamos la homografía
		homografias[i] = (findHomography(pointSrc, pointDst, CV_RANSAC, 1, noArray()));
		homografias[i] = homografias[i+1]*homografias[i];
	}	
	
	//Componemos el panorama
	warpPerspective(vecImg[0], canvas, homografias[0], Size(canvas.cols, canvas.rows), 1, BORDER_CONSTANT, colores[NEGRO]);
	for(int i=1; i<vecImg.size(); i++){ 
		//Calculamos los puntos para la región a mezclar
		p1=warpPoint(Point(0,0),homografias[i]);                    //Punto (0,0) de la img der
		p2=warpPoint(Point(vecImg[i-1].cols-1,0),homografias[i-1]); //Punto (cols, 0) de la img izq
		//Si la región sale demasiado grande, la acotamos
		if(p2.x-p1.x>vecImg[i].cols*0.20) regionComun = vecImg[i].cols*0.20;
		else				   			  regionComun = p2.x-p1.x;
		//Definimos la región para que sea divisible entre dos
		calcularPotDosCercana(regionComun, (regionComun)/4, maxLevels, ancho, levels);		

		//Transformamos la imagen a incluir en el panorama
		warpPerspective(vecImg[i], imgPers, homografias[i], Size(canvas.cols, canvas.rows), 1, BORDER_TRANSPARENT, colores[NEGRO]);

		//Definimos los ROI de cada imagen para mezclar
		roi1 = canvas (Rect(p1, Size(ancho, vecImg[i].rows)));  //Roi de la img anterior que está en el canvas
		roi2 = imgPers(Rect(p1, Size(ancho, vecImg[i].rows)));  //Roi de la img que hay que meter en el canvas
		
		cout <<"Imagen numero " << i << ". Niveles piramide: "<<levels << "\tAncho region mezclada: " << ancho << endl ;

		//Mezclamos ambas regiones y copiamos los resultados a la imagen a incluir en el panorama
		mezclaPiramidal(roi1,roi2, imgMezclada, levels);
		imgMezclada.copyTo(roi2);
		
		//Calculamos la área final, en la que va la imagen
		p3=warpPoint(Point(vecImg[i].cols-1,vecImg[i].rows-1),homografias[i]);
		roi1=imgPers(Rect(p1, p3));
		roi2=canvas(Rect(p1, p3));
		//Y la pegamos
		roi1.copyTo(roi2);
	}
	
	//Quitamos los bordes negros
	canvas=cortarBordes(canvas);
	
	return canvas;	
}

/*************************************************************************************************
 * Main                                                                                          *
 *************************************************************************************************/
int main ( int argc, char* argv[]){	
	Mat imgOut;
	vector<Mat> imgPano, imgPanoOut;
	float f;
	string inFolder;
	if(argc!=4 && argc != 3){
		cout << "Numero de argumentos incorrecto. El formato correcto es:" << endl << endl;
		cout << "\t" << argv[0] << " <directorio fotos> <factor de escalado> [<panorama compuesto jpg>]" << endl;
		cout << "Si no se especifica fichero de salida el resultado se mostrara por pantalla" << endl;
		cout << "ADVERTENCIA: Debido a las limitaciones de OpenCV, puede ser que en pantalla aparezca cortado el panorama" << endl;
		exit(1);
	}
	inFolder=argv[1];  //carpeta
	f=atof(argv[2]);   //Factor escala para las cilindricas

	//Cargamos las imágenes del directorio
	leerFicheros(inFolder,imgPano);	
	//Pasamos a cilíndricas
	imgACilindricas(imgPano, imgPanoOut, f);
	//Cortamos los bordes negros del lateral
	for(int i=0; i<imgPanoOut.size(); i++)
		imgPanoOut[i] = cortarLaterales(imgPanoOut[i]);
	//Construimos panorama
	imgOut = panorama(imgPanoOut);
	
	//Imprimimos por pantalla o guardamos el panorama en disco
	if(argc==3)	pintaImagen(imgOut,inFolder);
	else		imwrite(argv[3], imgOut);;
	
	return 0;
}
