#include <iostream>
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <vector>
#include "TCanvas.h"
#include "TLegend.h"
#include "TFile.h"


using namespace std;
using std::cout;
using std::endl;

void circle_detector1(){

  float px,py,a,b,r,x,y,y1,y2,x1,x2,x_2,y_2,x_1,y_1;
  int rd;
  float i,j,k,l;
  float q,B,pt;
  float w=1.0;
  int hpt,lpt;
  int ranchar;
  int event;

  //////// defining magnetic field
  B=1;

  //////// creating the random number generator
  TRandom *grandom = new TRandom3();
   
  /*
    pt = tranverse momentum in GeV
    r = radius of trajectory of the particle
    rd = radius of detector layer
    x,y = coordinates of intersection point
    hpt = high pt track
    lpt = low pt track
   */

  fstream file;
  file.open("event_data.txt", ios::out);
  
  for(event = 1 ; event<20001;event++){
    int ntrack, notrk;
    ntrack = 100;   //choose total number of tracks
    hpt = 1;      //choose number of high pt tracks
    lpt = ntrack-hpt; //numbe of low pt tracks
    
    /////// for low pt tracks
    for( int trk=1; trk < hpt+1; trk++){
      
      //for charge//
      ranchar = grandom->Uniform(1,3);
      if(ranchar < 2.0){
	q=1;
      }
      else{
	q=-1;
      }
      
      //////////////////
      //for high track//
      //////////////////

      
      pt = grandom->Uniform(100,150);
      px = grandom->Uniform(-pt,pt);
      py = pow(pt*pt-px*px,0.5);
      ranchar = grandom->Uniform(1,3);
      if(ranchar==1){
	py=py;
      }
      else{
	py=-py;
      }
      
      
      r = pow((px*px+py*py),0.5)/abs(q*B); 
      a = -py/abs(q*B);
      b = px/abs(q*B);
      
      if(abs(r)>10){
	
	for(rd=5;rd<21;rd++){
	  float temp = pow(((4*r*r)-(rd*rd)),0.5);
	  y1 = ((b*rd*rd)/(2*r*r)) + (((a*rd)/(2*r*r))*temp); //clockwise
	  y2 = ((b*rd*rd)/(2*r*r)) - (((a*rd)/(2*r*r))*temp); //counter clockwise
	  x1 = ((rd*rd)/(2*a))-(b*y1/a);
	  x2 = ((rd*rd)/(2*a))-(b*y2/a);
	  
	  if(q==1){
	    for(int j=1;;j++){
	      i =  grandom->Uniform(-1000,1000)+(x2*1000);
	      x_2 = i/1000;
	      if(abs(x_2)<rd){break;}
	    }
	    if(y2<0){
	      y_2 = -sqrt(pow(rd,2)-pow(x_2,2));}
	    else{
	      y_2 = sqrt(pow(rd,2)-pow(x_2,2));}
	    
	    file <<fixed<<setprecision(2)<< x_2 <<"\t"<< y_2 <<"\t";
	  }
	  
	  if(q==-1){
	    for(int j=1;;j++){
	      i =  grandom->Uniform(-1000,1000)+(x1*1000);
	      x_1 = i/1000;
	      if(abs(x_1)<rd){break;}
	    }
	    if(y1>0){
	      y_1 = sqrt(pow(rd,2)-pow(x_1,2));}
	    else{
	      y_1 = -sqrt(pow(rd,2)-pow(x_1,2));}
	    
	    file <<fixed<<setprecision(2)<< x_1 <<"\t"<< y_1<<"\t";
	  }
	}	
      }
    }
    
    for(int trk=1; trk<lpt+1 ;trk++){
      
      //for charge//
      ranchar = grandom->Uniform(1,3);
      if(ranchar < 2.0){
	q=1;
      }
      else{
	q=-1;
      }
      
      ////////////////////
      //for low pt track//
      ////////////////////
      
      pt = grandom->Uniform(10.01,50);
      px = grandom->Uniform(-pt,pt);
      py = pow(pt*pt-px*px,0.5);
      ranchar = grandom->Uniform(1,3);
      if(ranchar==1){
	py=py;
      }
      else{
	py=-py;
      }
      
      
      r = pow((px*px+py*py),0.5)/abs(q*B); 
      a = -py/abs(q*B);
      b = px/abs(q*B);
      
      if(abs(r)>10){
	
	for(rd=5;rd<21;rd++){
	  float temp = pow(((4*r*r)-(rd*rd)),0.5);
	  y1 = ((b*rd*rd)/(2*r*r)) + (((a*rd)/(2*r*r))*temp); //clockwise
	  y2 = ((b*rd*rd)/(2*r*r)) - (((a*rd)/(2*r*r))*temp); //counter clockwise
	  x1 = ((rd*rd)/(2*a))-(b*y1/a);
	  x2 = ((rd*rd)/(2*a))-(b*y2/a); // both equation have negative sign here just y values are different.
	  
	  if(q==1){
	    for(int j=1;;j++){
	      i =  grandom->Uniform(-1000,1000)+(x2*1000);
	      x_2 = i/1000;
	      if(abs(x_2)<rd){break;}
	    }
	    if(y2<0){
	      y_2 = -sqrt(pow(rd,2)-pow(x_2,2));}
	    else{
	      y_2 = sqrt(pow(rd,2)-pow(x_2,2));}
	    
	    file <<fixed<<setprecision(2)<< x_2 <<"\t"<< y_2 <<"\t";
	  }
	  
	  if(q==-1){
	    for(int j=1;;j++){
	      i =  grandom->Uniform(-1000,1000)+(x1*1000);
	      x_1 = i/1000;
	      if(abs(x_1)<rd){break;}
	    }
	    if(y1>0){
	      y_1 = sqrt(pow(rd,2)-pow(x_1,2));}
	    else{
	      y_1 = -sqrt(pow(rd,2)-pow(x_1,2));}
	    
	    file <<fixed<<setprecision(2)<< x_1 <<"\t"<< y_1<<"\t";
	  }
	}	
      }
    }
    file <<endl;
  } 
}


