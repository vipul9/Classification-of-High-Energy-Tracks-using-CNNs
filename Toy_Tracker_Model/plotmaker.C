#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <stdio.h>
#include <math.h>
#include <vector>
#include "TGraph.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TFile.h"
#include "TROOT.h"


using namespace std;
using std::cout;
using std::endl;

void plotmaker(){
  
  gROOT->SetBatch(kTRUE);
  gErrorIgnoreLevel = 20001;
  
  float nentries;
 
  Float_t ntrk,trk,event;
  Float_t x[1601],y[1601];
  TFile* file = new TFile("training_testing.root","read");
  TTree *tree = (TTree*)file->Get("TrackHit"); 
  

  int n =1;
  for(int i=1;i<1601;i++){
    ostringstream branch_num;
    branch_num << i;
    string temp_x = "x" + branch_num.str();
    string temp_y = "y" + branch_num.str();
    char* xbranch_addr = (char*) temp_x.c_str();
    char* ybranch_addr = (char*) temp_y.c_str();
    //cout << xbranch_addr << "\t" << ybranch_addr << "\n";
    
    tree->SetBranchAddress(xbranch_addr,&x[i-1]);
    tree->SetBranchAddress(ybranch_addr,&y[i-1]);
  }
  
  nentries = tree->GetEntries();
  
  for(int event=0; event<nentries; event++){
    
    tree->GetEntry(event);
    
    TCanvas *c1 = new TCanvas("c1","xy",512+4,512+28);
    
    TGraph *gr = new TGraph(1600,x,y);
    gr->GetYaxis()->SetRangeUser(-25.0,25.0); //only works for y axis
    gr->GetXaxis()->SetLimits(-25.0,25.0); // only works for x axis
    gr->SetMarkerStyle(8);
    gr->SetMarkerSize(0.5);
    gr->SetTitle(" ");
    gr->Draw("AP");
    
    ostringstream str1;
    
    // Sending a number as a stream into output 
    // string 
    str1 << event;
    
    // the str() coverts number into string 
    string temp_iname = "img"+str1.str()+".png"; 
    char* img_name=(char*) temp_iname.c_str();
    
    c1->SaveAs(img_name);
    c1->Clear();
    
    if(event%100 == 0){
      cout << "number of images created = " << event << "\n";
      cout.flush();
    } 
  }
}


