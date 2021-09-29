#include "CamCal.h"
#include "utils.h"
#include <string>
using namespace cv;
C2dPtSel o2dPtSel;

void on_mouse(int event, int x, int y, int flags, void*)  // mouse event
{
	if (!o2dPtSel.chkImgLd())
	{
		std::cout << "Error: on_mouse(): frame image is unloaded" << std::endl;
		return;
	}

	if (event == EVENT_FLAG_LBUTTON)
		o2dPtSel.addNd(x, y);

	return;
}

CCamCal::CCamCal(void)
{
	// list of 3D points for PnP
	std::vector<cv::Point2f>().swap(m_vo3dPt);

	// list of 2D points for PnP
	std::vector<cv::Point2f>().swap(m_vo2dPt);
}

CCamCal::~CCamCal(void)
{
	// list of 3D points for PnP
	std::vector<cv::Point2f>().swap(m_vo3dPt);

	// list of 2D points for PnP
	std::vector<cv::Point2f>().swap(m_vo2dPt);
}

void CCamCal::initialize(CCfg oCfg, cv::Mat oImgFrm)
{
	// configuration parameters
	m_oCfg = oCfg;

	// frame image
	m_oImgFrm = oImgFrm.clone();

	// list of 3D points for PnP
	m_vo3dPt = m_oCfg.getCal3dPtLs();

	// list of 2D points for PnP
	if (!m_oCfg.getCalSel2dPtFlg())
		m_vo2dPt = m_oCfg.getCal2dPtLs();

	// homography matrix
	m_oHomoMat = cv::Mat(3, 3, CV_64F);

	// reprojection error
	m_fReprojErr = DBL_MAX;
}

void CCamCal::process(void)
{
	// select 2D points if they are not provided in the configuration file
	if (m_oCfg.getCalSel2dPtFlg())
	{
		std::vector<cv::Point2f>().swap(m_vo2dPt);
		o2dPtSel.initialize(m_oCfg, m_oImgFrm);
		std::vector<cv::Point> vo2dPt = o2dPtSel.process();
		std::cout << "Selected 2D points on the frame image: " << std::endl;
		for (int i = 0; i < vo2dPt.size(); i++)
		{
			m_vo2dPt.push_back(vo2dPt[i]);
			if ((vo2dPt.size() - 1) > i)
				std::cout << "[ " << vo2dPt[i].x << ", " << vo2dPt[i].y << " ]," << std::endl;
			else
				std::cout << "[ " << vo2dPt[i].x << ", " << vo2dPt[i].y << " ]" << std::endl;
		}
	}
	
	if(!m_oCfg.withEdaOpt()){
		
		// compute homography matrix
		if (-1 == m_oCfg.getCalTyp()){
			// run all calibration types
			runAllCalTyp(m_vo3dPt, m_vo2dPt);
		}
		else
		{	
			m_oHomoMat = cv::findHomography(m_vo3dPt, m_vo2dPt, m_oCfg.getCalTyp(), m_oCfg.getCalRansacReprojThld());
			m_fReprojErr = calcReprojErr(m_vo3dPt, m_vo2dPt, m_oHomoMat, m_oCfg.getCalTyp(), m_oCfg.getCalRansacReprojThld(), "2D3D");
			m_projErr = calcReprojErr(m_vo3dPt, m_vo2dPt, m_oHomoMat, m_oCfg.getCalTyp(), m_oCfg.getCalRansacReprojThld(), "3D2D");
			
		}
			dReprojErr = calcDistReprojErr(m_vo3dPt, m_vo2dPt, m_oHomoMat, m_oCfg.getCalTyp(), m_oCfg.getCalRansacReprojThld());
			//change from Km to m
			dReprojErr *= 1000;
	}
	else{
		calCamEdaOpt();
		m_fReprojErr = calcReprojErr(m_vo3dPt, m_vo2dPt, m_oHomoMat, m_oCfg.getCalTyp(), m_oCfg.getCalRansacReprojThld(), "2D3D");
		m_projErr = calcReprojErr(m_vo3dPt, m_vo2dPt, m_oHomoMat, m_oCfg.getCalTyp(), m_oCfg.getCalRansacReprojThld(), "3D2D");
		dReprojErr = calcDistReprojErr(m_vo3dPt, m_vo2dPt, m_oHomoMat, m_oCfg.getCalTyp(), m_oCfg.getCalRansacReprojThld());
		//change from Km to m
		dReprojErr *= 1000;
	}

	std::cout << std::endl;
}

bool compDistError(PtObj oCamParam1, PtObj oCamParam2)
{
	return (oCamParam1.getProjErr() < oCamParam2.getProjErr());
}

std::vector<cv::Point2f> CCamCal::initPts(PtObj sPtParamsObj){
	double pt3dx, pt3dy, pt3dx, pt3dy;
	std::vector<cv::Point2f> randPt3dVec;

	for(int i = 0; i < sPtParamsObj.pt2dVecMin.size(); i++){
		
		pt3dx = get_rand_num(sPtParamsObj.pt3dVecMin[i].x, sPtParamsObj.pt3dVecMax[i].x, rand());
		pt3dy = get_rand_num(sPtParamsObj.pt3dVecMin[i].y, sPtParamsObj.pt3dVecMax[i].y, rand());
		
		randPt2dVec.push_back({pt3dx, pt3dy});
	}
	return randPt3dVec;
}

PtObj CCamCal::initEdaParamRng(std::vector<cv::Point2f> m_vo3dPt){
	PtObj sParamRng;

	if(m_vo3dPt.size() != m_vo2dPt.size())
		std::cout << "Sizes should be equal" << std::endl;
	for(int i = 0; i < m_vo3dPt.size(); i++){

		sParamRng.pt3dVecMax.push_back({m_vo3dPt[i].x  + m_oImgFrm.cols*EDA_RNG_2DPT, m_vo3dPt[i].y  + m_oImgFrm.rows*EDA_RNG_2DPT});
		sParamRng.pt3dVecMin.push_back({m_vo3dPt[i].x  - m_oImgFrm.cols*EDA_RNG_2DPT, m_vo3dPt[i].y  - m_oImgFrm.rows*EDA_RNG_2DPT});
		
	}
	sParamRng.setVectorReady();
	return sParamRng;
}

PtObj CCamCal::estEdaParamRng(std::vector<PtObj>* pvoPtParams){
	int nCamParamNum = pvoPtParams->size(), nParamNum = pvoPtParams->begin()->getSize()*2, iParam, iCamParam = 0;
	double fParamVar;
	double* afParamMean = (double*)calloc(nParamNum, sizeof(double));
	double* afParamData = (double*)calloc(nParamNum*nCamParamNum, sizeof(double));

	std::vector<PtObj>::iterator ivoPtParams;
	PtObj sParamRng;

	for (ivoPtParams = pvoPtParams->begin(); ivoPtParams != pvoPtParams->end(); ivoPtParams++){
		iParam = 0;
		for(int i = 0; i < ivoPtParams->getSize(); i++){
			afParamData[iParam*nCamParamNum + iCamParam] = ivoPtParams->getRand3dPt(i).x; afParamMean[2*i+0] += afParamData[iParam*nCamParamNum + iCamParam]; iParam++;
			afParamData[iParam*nCamParamNum + iCamParam] = ivoPtParams->getRand3dPt(i).y; afParamMean[2*i+1] += afParamData[iParam*nCamParamNum + iCamParam]; iParam++;
		}
		iCamParam++;
	}
	for (iParam = 0; iParam < nParamNum; iParam++)
		afParamMean[iParam] /= nCamParamNum;

	double pt3dxmax, pt3dymax, pt3dxmin, pt3dymin;
	for(int i = 0; i < ivoPtParams->getSize(); i++){
		// 2dpt x
		iParam = 0 + 2*i;
		fParamVar = 0.0f;
		for(iCamParam = 0; iCamParam < nCamParamNum; iCamParam++){
			fParamVar += (afParamData[(iParam * nCamParamNum) + iCamParam] - afParamMean[iParam]) *
			(afParamData[(iParam * nCamParamNum) + iCamParam] - afParamMean[iParam]);
		}
		fParamVar /= nCamParamNum;
		pt3dxmax = afParamMean[iParam] + std::sqrt(fParamVar);
		pt3dxmin = afParamMean[iParam] - std::sqrt(fParamVar);
		// 2dpt y
		iParam = 1 + 2*i;
		fParamVar = 0.0f;
		for(iCamParam = 0; iCamParam < nCamParamNum; iCamParam++){
			fParamVar += (afParamData[(iParam * nCamParamNum) + iCamParam] - afParamMean[iParam]) *
			(afParamData[(iParam * nCamParamNum) + iCamParam] - afParamMean[iParam]);
		}
		fParamVar /= nCamParamNum;
		pt3dymax = afParamMean[iParam] + std::sqrt(fParamVar);
		pt3dymin = afParamMean[iParam] - std::sqrt(fParamVar);

		sParamRng.pt3dVecMin.push_back({pt3dxmin, pt3dymin});
		sParamRng.pt3dVecMax.push_back({pt3dxmax, pt3dymax});

	}
	sParamRng.setVectorReady();
	std::free(afParamMean);
	std::free(afParamData);
	
	return sParamRng;
}

void CCamCal::calCamEdaOpt(void){
	
	int nR = EDA_INIT_POP, nN = EDA_SEL_POP, nIterNum = EDA_ITER_NUM, iIter = 0, iProc;
    bool bProc25, bProc50, bProc75;
	
	double fReprojErr, fReprojErrMean, fReprojErrMeanPrev, fReprojErrStd;
	double projErr, projErrMean, projErrMeanPrev, projErrStd;
	double dReprojErr, dReprojErrMean, dReprojErrMeanPrev, dReprojErrStd;
	
	
	
	cv::Mat oHomoMat;

	PtObj sPtParamsObj;
	//initialize range of points
	
	sPtParamsObj = initEdaParamRng(m_vo3dPt);
	
	// EDA optimization
	if(nN >= nR)
		std::printf("Error: Selected population should be less than initial population\n");
	std::vector<PtObj> voPtParams;
	std::vector<PtObj>::iterator ivoPtParams;
	
	for(int iR = 0; iR < nR; iR++){
		
		sPtParamsObj.randPt3dVec = initPts(sPtParamsObj);
		voPtParams.push_back(sPtParamsObj);
	}

	std::printf("Start EDA optimization for camera calibration\n");
	while (nIterNum > iIter){
		printf("==== generation %d: ====\n", iIter);
		iProc = 0;
		bProc25 = false;
		bProc50 = false;
		bProc75 = false;
		fReprojErrMean = 0.0;
		projErrMean = 0.0;
		fReprojErrStd = 0.0;
		projErrStd = 0.0;

		for(ivoPtParams = voPtParams.begin(); ivoPtParams != voPtParams.end(); ivoPtParams++){
			std::vector<::Point2f> curr3dPtSet;
			for( int i = 0; i < m_vo3dPt.size(); i++){
				curr3dPtSet.push_back(ivoPtParams->getRand3dPt(i));
			}
			// compute homography matrix
			if (-1 == m_oCfg.getCalTyp()){
				// run all calibration types
				
				runAllCalTyp(curr3dPtSet, m_vo2dPt);
			}
			else{

				m_oHomoMat = cv::findHomography(curr3dPtSet, m_vo2dPt, m_oCfg.getCalTyp(), m_oCfg.getCalRansacReprojThld());
				m_fReprojErr = calcReprojErr(curr3dPtSet, m_vo2dPt, m_oHomoMat, m_oCfg.getCalTyp(), m_oCfg.getCalRansacReprojThld(), "2D3D");
				m_projErr = calcReprojErr(curr3dPtSet, m_vo2dPt, m_oHomoMat, m_oCfg.getCalTyp(), m_oCfg.getCalRansacReprojThld(), "3D2D");
			}
			m_vo3dPt = curr3dPtSet;
			ivoPtParams->setHomoMat(m_oHomoMat);
			dReprojErr = calcDistReprojErr(m_vo3dPt, m_vo2dPt, m_oHomoMat, m_oCfg.getCalTyp(), m_oCfg.getCalRansacReprojThld());

			dReprojErr *= 1000;

			ivoPtParams->setReprojErr(m_fReprojErr);
			ivoPtParams->setProjErr(m_projErr);
			ivoPtParams->setDistReprojErr(dReprojErr);

			fReprojErrMean += m_fReprojErr;
			projErrMean += m_projErr;
			dReprojErrMean += dReprojErr;
			iProc++;

			if ((((float)iProc / (float)nR) > 0.25) && (!bProc25)) { std::printf("25%%..."); bProc25 = true; }
			if ((((float)iProc / (float)nR) > 0.50) && (!bProc50)) { std::printf("50%%..."); bProc50 = true; }
			if ((((float)iProc / (float)nR) > 0.75) && (!bProc75)) { std::printf("75%%..."); bProc75 = true; }
		}

		fReprojErrMean /= nR;
		projErrMean /= nR;
		dReprojErrMean /= nR; 

		for(ivoPtParams = voPtParams.begin(); ivoPtParams != voPtParams.end(); ivoPtParams++){
			double fReprojErr = ivoPtParams->getReprojErr();
			double projErr = ivoPtParams->getProjErr();
			double dReprojErr = ivoPtParams->getDistReprojErr();
			fReprojErrStd += (fReprojErr - fReprojErrMean) * (fReprojErr - fReprojErrMean);
			projErrStd += (projErr - projErrMean) * (projErr - projErrMean);
			dReprojErrStd += (dReprojErr - dReprojErrMean) * (dReprojErr - dReprojErrMean);
		}
		fReprojErrStd = std::sqrt(fReprojErrStd / nR);
		projErrStd = std::sqrt(projErrStd / nR);
		dReprojErrStd = std::sqrt(dReprojErrStd / nR);

		std::printf("100%%!\n");
		std::printf("current reprojection error mean = %f\n", fReprojErrMean);
		std::printf("current projection error mean = %f\n", projErrMean);
		std::printf("current distances error mean = %f\n", dReprojErrMean);
		std::printf("current reprojection error standard deviation = %f\n", fReprojErrStd);
		std::printf("current projection error standard deviation = %f\n", projErrStd);
		std::printf("current distances error standard deviation = %f\n", dReprojErrStd);
		
		if(!fReprojErrMean || !dReprojErrMean){
			std::printf("Camera calibration failed. \n");
			break;
		}


		fReprojErrMeanPrev = fReprojErrMean;
		projErrMeanPrev = projErrMean;
		dReprojErrMeanPrev = dReprojErrMean;

		std::stable_sort(voPtParams.begin(), voPtParams.end(), compDistError);

		voPtParams.erase(voPtParams.begin() + nN, voPtParams.end());

		//check if generation needs to stop

		if((0 < iIter) && ((projErrMeanPrev * EDA_REPROJ_ERR_THLD) > std::abs(projErrMean - projErrMeanPrev))){
			std::printf("Projection error is small enough. Stop generation.\n");
			break;
		}

		sPtParamsObj = estEdaParamRng(&voPtParams);

		for(int iR = 0; iR < nR; iR++){
			sPtParamsObj.randPt3dVec = initPts(sPtParamsObj);
			voPtParams.push_back(sPtParamsObj);
		}
		iIter++;
	}

	m_vo3dPt = voPtParams[0].randPt3dVec;

	if(nIterNum <= iIter){
		printf("Exit: Results can not converge.\n");
	}

}

void CCamCal::output(void)
{
	// output text file of homography matrix
	outTxt();

	// plot a display grid on the ground plane
	pltDispGrd();
}

void CCamCal::runAllCalTyp(std::vector<cv::Point2f> vo3dPt, std::vector<cv::Point2f> vo2dPt)
{
	cv::Mat oHomoMat;
	double fReprojErr, fProjErr;

	// a regular method using all the points
	try
	{
		oHomoMat = cv::findHomography(vo3dPt, vo2dPt, 0, 0);
		fReprojErr = calcReprojErr(vo3dPt, vo2dPt, oHomoMat, 0, 0, "3D2D");
		fProjErr = calcReprojErr(vo3dPt, vo2dPt, oHomoMat, 0, 0, "2D3D");

		if (fProjErr < m_projErr)
		{
        		m_fReprojErr = fReprojErr;
        		m_oHomoMat = oHomoMat;
				m_projErr = fProjErr;
		}
        }
        catch(cv::Exception& e)
        {
                const char* pcErrMsg = e.what();
                std::cout << "Exception caught: " << pcErrMsg << std::endl;
        }

	// Least-Median robust method
	try
	{
		oHomoMat = cv::findHomography(vo3dPt, vo2dPt, 4, 0);
		fReprojErr = calcReprojErr(vo3dPt, vo2dPt, oHomoMat, 4, 0, "3D2D");
		fProjErr = calcReprojErr(vo3dPt, vo2dPt, oHomoMat, 4, 0, "2D3D");

		if (fProjErr < m_projErr)
		{
        		m_fReprojErr = fReprojErr;
        		m_oHomoMat = oHomoMat;
				m_projErr = fProjErr;
		}
        }
        catch(cv::Exception& e)
        {
                const char* pcErrMsg = e.what();
                std::cout << "Exception caught: " << pcErrMsg << std::endl;
        }

	// RANSAC-based robust method
	for (double t = 100; t >= 10; t -= 5)
	{
		try
		{
			oHomoMat = cv::findHomography(vo3dPt, vo2dPt, 8, t);
			fReprojErr = calcReprojErr(vo3dPt, vo2dPt, oHomoMat, 8, t, "3D2D");
			fProjErr = calcReprojErr(vo3dPt, vo2dPt, oHomoMat, 8, t, "2D3D");
			if (fProjErr < m_projErr)
        		{
        			m_fReprojErr = fReprojErr;
        			m_oHomoMat = oHomoMat;
					m_projErr = fProjErr;
        		}
        	}
        	catch(cv::Exception& e)
        	{
                	const char* pcErrMsg = e.what();
                	std::cout << "Exception caught: " << pcErrMsg << std::endl;
        	}
	}
	
}

double CCamCal::calcDistReprojErr(std::vector<cv::Point2f> vo3dPt, std::vector<cv::Point2f> vo2dPt, cv::Mat oHomoMat, int nCalTyp, double fCalRansacReprojThld){
	
	std::vector<cv::Point2f> backProjPts;
	for(int i = 0; i < vo2dPt.size(); i++){
		backProjPts.push_back(backproj2D3D(vo2dPt[i], oHomoMat));
	}

	std::vector<std::vector<cv::Point2f>> pairs3D = gen_pairs(vo3dPt);

	std::vector<std::vector<cv::Point2f>> pairsBackproj = gen_pairs(backProjPts);

	double predDist;
	double realDist;
	double distReprojErr = 0;

	if(pairs3D.size() != pairsBackproj.size()) printf("Not equal sizes, when calculating dist error\n");
	for( int i = 0; i < pairs3D.size(); i++){
		realDist = haversine_distance(pairs3D[i][0], pairs3D[i][1]);
		predDist = haversine_distance(pairsBackproj[i][0], pairsBackproj[i][1]);
		distReprojErr += pow(realDist - predDist, 2);
	}

	//RMSE root mean square error
	distReprojErr /= pairs3D.size();
	distReprojErr = std::sqrt(distReprojErr);
/*
	if (8 == nCalTyp)
		std::cout << "RMSE distance reprojection error of method #" << nCalTyp << " (threshold: " << fCalRansacReprojThld << "): " << distReprojErr << std::endl;
	else
		std::cout << "RMSE distance reprojection error of method #" << nCalTyp << ": " << distReprojErr << std::endl;
*/
    return distReprojErr;
	
}

double CCamCal::calcReprojErr3D(std::vector<cv::Point2f> vo3dPt, std::vector<cv::Point2f> vo2dPt, cv::Mat oHomoMat, int nCalTyp, double fCalRansacReprojThld){
	
	double fReprojErr = 0;

	for(int i = 0; i < vo3dPt.size(); i++){
		cv::Point2f pt3d;
		pt3d = backproj2D3D(vo2dPt[i], oHomoMat);
		fReprojErr += cv::norm(vo3dPt[i] - pt3d);
	}

	fReprojErr /= vo3dPt.size();
	return fReprojErr;
}

double CCamCal::calcReprojErr2D(std::vector<cv::Point2f> vo3dPt, std::vector<cv::Point2f> vo2dPt, cv::Mat oHomoMat, int nCalTyp, double fCalRansacReprojThld)
{
	double fReprojErr = 0;

	for (int i = 0; i < vo3dPt.size(); i++)
	{
		cv::Point2f o2dPt;
		o2dPt = proj3D2D(vo3dPt[i], oHomoMat);
		fReprojErr += cv::norm(vo2dPt[i] - o2dPt);
	}

	fReprojErr /= vo3dPt.size();
/*
	if (8 == nCalTyp)
		std::cout << "Average reprojection error of method #" << nCalTyp << " (threshold: " << fCalRansacReprojThld << "): " << fReprojErr << std::endl;
	else
		std::cout << "Average reprojection error of method #" << nCalTyp << ": " << fReprojErr << std::endl;
*/
    return fReprojErr;
}

double CCamCal::calcReprojErr(std::vector<cv::Point2f> vo3dPt, std::vector<cv::Point2f> vo2dPt, cv::Mat oHomoMat, int nCalTyp, double fCalRansacReprojThld, std::string mode){

	if(mode == "2D3D") return calcReprojErr3D(vo3dPt, vo2dPt, oHomoMat, nCalTyp, fCalRansacReprojThld);
	else if (mode =="3D2D") return calcReprojErr2D(vo3dPt, vo2dPt, oHomoMat, nCalTyp, fCalRansacReprojThld);
	else throw "Invalid Option. You should specify wether 2D3D or 3D2D";
}

void CCamCal::outTxt(void)
{
	FILE* pfHomoMat = std::fopen(m_oCfg.getOutCamMatPth(), "w");

	std::fprintf(pfHomoMat, "Homography matrix: %.15lf %.15lf %.15lf;%.15lf %.15lf %.15lf;%.15lf %.15lf %.15lf\n",
		m_oHomoMat.at<double>(0, 0), m_oHomoMat.at<double>(0, 1), m_oHomoMat.at<double>(0, 2),
		m_oHomoMat.at<double>(1, 0), m_oHomoMat.at<double>(1, 1), m_oHomoMat.at<double>(1, 2),
		m_oHomoMat.at<double>(2, 0), m_oHomoMat.at<double>(2, 1), m_oHomoMat.at<double>(2, 2));
	std::printf("Homography matrix: %.15lf %.15lf %.15lf;%.15lf %.15lf %.15lf;%.15lf %.15lf %.15lf\n",
		m_oHomoMat.at<double>(0, 0), m_oHomoMat.at<double>(0, 1), m_oHomoMat.at<double>(0, 2),
		m_oHomoMat.at<double>(1, 0), m_oHomoMat.at<double>(1, 1), m_oHomoMat.at<double>(1, 2),
		m_oHomoMat.at<double>(2, 0), m_oHomoMat.at<double>(2, 1), m_oHomoMat.at<double>(2, 2));

	if (m_oCfg.getCalDistFlg())
	{
	    cv::Mat oCalIntMat = m_oCfg.getCalIntMat();
	    std::fprintf(pfHomoMat, "Intrinsic parameter matrix: %.15lf %.15lf %.15lf;%.15lf %.15lf %.15lf;%.15lf %.15lf %.15lf\n",
            	oCalIntMat.at<double>(0, 0), oCalIntMat.at<double>(0, 1), oCalIntMat.at<double>(0, 2),
            	oCalIntMat.at<double>(1, 0), oCalIntMat.at<double>(1, 1), oCalIntMat.at<double>(1, 2),
            	oCalIntMat.at<double>(2, 0), oCalIntMat.at<double>(2, 1), oCalIntMat.at<double>(2, 2));
	    std::printf("Intrinsic parameter matrix: %.15lf %.15lf %.15lf;%.15lf %.15lf %.15lf;%.15lf %.15lf %.15lf\n",
            	oCalIntMat.at<double>(0, 0), oCalIntMat.at<double>(0, 1), oCalIntMat.at<double>(0, 2),
            	oCalIntMat.at<double>(1, 0), oCalIntMat.at<double>(1, 1), oCalIntMat.at<double>(1, 2),
            	oCalIntMat.at<double>(2, 0), oCalIntMat.at<double>(2, 1), oCalIntMat.at<double>(2, 2));

		cv::Mat oCalDistCoeffMat = m_oCfg.getCalDistCoeffMat();
		std::fprintf(pfHomoMat, "Distortion coefficients: %.15lf %.15lf %.15lf %.15lf\n",
			oCalDistCoeffMat.at<double>(0), oCalDistCoeffMat.at<double>(1),
			oCalDistCoeffMat.at<double>(2), oCalDistCoeffMat.at<double>(3));
		std::printf("Distortion coefficients: %.15lf %.15lf %.15lf %.15lf\n",
			oCalDistCoeffMat.at<double>(0), oCalDistCoeffMat.at<double>(1),
			oCalDistCoeffMat.at<double>(2), oCalDistCoeffMat.at<double>(3));
	}

	std::fprintf(pfHomoMat, "Projection error: %.15lf\n", m_projErr);
	std::printf("Projection error: %.15lf\n", m_projErr);
	std::fprintf(pfHomoMat, "Backrojection error: %.15lf\n", m_fReprojErr);
	std::printf("Backprojection error: %.15lf\n", m_fReprojErr);
	std::fprintf(pfHomoMat, "Distance error: %.15lf\n", dReprojErr);
	std::printf("Distance error: %.15lf\n", dReprojErr);

	std::fclose(pfHomoMat);
}

void CCamCal::pltDispGrd(void)
{
	cv::Mat oImgPlt = m_oImgFrm.clone();
	cv::Size oDispGrdDim = m_oCfg.getCalDispGrdDim();

	// find the limits of the 3D grid on the ground plane
	double fXMin = DBL_MAX;
	double fYMin = DBL_MAX;
	double fXMax = -DBL_MAX;
	double fYMax = -DBL_MAX;

	for (int i = 0; i < m_vo3dPt.size(); i++)
	{
		if (fXMin > m_vo3dPt[i].x)
			fXMin = m_vo3dPt[i].x;

		if (fYMin > m_vo3dPt[i].y)
			fYMin = m_vo3dPt[i].y;

		if (fXMax < m_vo3dPt[i].x)
			fXMax = m_vo3dPt[i].x;

		if (fYMax < m_vo3dPt[i].y)
			fYMax = m_vo3dPt[i].y;
	}

	// compute the endpoints for the 3D grid on the ground plane
	std::vector<cv::Point2f> vo3dGrdPtTop, vo3dGrdPtBtm, vo3dGrdPtLft, vo3dGrdPtRgt;

	for (int x = 0; x < oDispGrdDim.width; x++)
	{
		vo3dGrdPtTop.push_back(cv::Point2f((fXMin + (x * ((fXMax - fXMin) / (oDispGrdDim.width - 1)))), fYMin));
		vo3dGrdPtBtm.push_back(cv::Point2f((fXMin + (x * ((fXMax - fXMin) / (oDispGrdDim.width - 1)))), fYMax));
	}

	for (int y = 0; y < oDispGrdDim.height; y++)
	{
		vo3dGrdPtLft.push_back(cv::Point2f(fXMin, (fYMin + (y * ((fYMax - fYMin) / (oDispGrdDim.height - 1))))));
		vo3dGrdPtRgt.push_back(cv::Point2f(fXMax, (fYMin + (y * ((fYMax - fYMin) / (oDispGrdDim.height - 1))))));
	}

	// compute the endpoints for the projected 2D grid
	std::vector<cv::Point2f> vo2dGrdPtTop, vo2dGrdPtBtm, vo2dGrdPtLft, vo2dGrdPtRgt;

	for (int i = 0; i < oDispGrdDim.width; i++)
	{
		cv::Mat o3dPtMat(3, 1, CV_64F);
		cv::Mat o2dPtMat(3, 1, CV_64F);

		o3dPtMat.at<double>(0, 0) = vo3dGrdPtTop[i].x;
		o3dPtMat.at<double>(1, 0) = vo3dGrdPtTop[i].y;
		o3dPtMat.at<double>(2, 0) = 1;
		o2dPtMat = m_oHomoMat * o3dPtMat;
		vo2dGrdPtTop.push_back(cv::Point2f((o2dPtMat.at<double>(0, 0) / o2dPtMat.at<double>(2, 0)), (o2dPtMat.at<double>(1, 0) / o2dPtMat.at<double>(2, 0))));

		o3dPtMat.at<double>(0, 0) = vo3dGrdPtBtm[i].x;
		o3dPtMat.at<double>(1, 0) = vo3dGrdPtBtm[i].y;
		o3dPtMat.at<double>(2, 0) = 1;
		o2dPtMat = m_oHomoMat * o3dPtMat;
		vo2dGrdPtBtm.push_back(cv::Point2f((o2dPtMat.at<double>(0, 0) / o2dPtMat.at<double>(2, 0)), (o2dPtMat.at<double>(1, 0) / o2dPtMat.at<double>(2, 0))));
	}

	for (int i = 0; i < oDispGrdDim.height; i++)
	{
		cv::Mat o3dPtMat(3, 1, CV_64F);
		cv::Mat o2dPtMat(3, 1, CV_64F);

		o3dPtMat.at<double>(0, 0) = vo3dGrdPtLft[i].x;
		o3dPtMat.at<double>(1, 0) = vo3dGrdPtLft[i].y;
		o3dPtMat.at<double>(2, 0) = 1;
		o2dPtMat = m_oHomoMat * o3dPtMat;
		vo2dGrdPtLft.push_back(cv::Point2f((o2dPtMat.at<double>(0, 0) / o2dPtMat.at<double>(2, 0)), (o2dPtMat.at<double>(1, 0) / o2dPtMat.at<double>(2, 0))));

		o3dPtMat.at<double>(0, 0) = vo3dGrdPtRgt[i].x;
		o3dPtMat.at<double>(1, 0) = vo3dGrdPtRgt[i].y;
		o3dPtMat.at<double>(2, 0) = 1;
		o2dPtMat = m_oHomoMat * o3dPtMat;
		vo2dGrdPtRgt.push_back(cv::Point2f((o2dPtMat.at<double>(0, 0) / o2dPtMat.at<double>(2, 0)), (o2dPtMat.at<double>(1, 0) / o2dPtMat.at<double>(2, 0))));
	}

	// draw grid lines on the frame image
	for (int i = 0; i < oDispGrdDim.width; i++)
		cv::line(oImgPlt, vo2dGrdPtTop[i], vo2dGrdPtBtm[i], cv::Scalar(int(255.0 * ((double)i / (double)oDispGrdDim.width)), 127, 127), 2, LINE_AA);

	for (int i = 0; i < oDispGrdDim.width; i++)
		cv::line(oImgPlt, vo2dGrdPtLft[i], vo2dGrdPtRgt[i], cv::Scalar(127, 127, int(255.0 * ((double)i / (double)oDispGrdDim.width))), 2, LINE_AA);

	// plot the 2D points
	for (int i = 0; i < m_vo2dPt.size(); i++)
	{
		char acPtIdx[32];
		cv::circle(oImgPlt, m_vo2dPt[i], 6, cv::Scalar(255, 0, 0), 1, LINE_AA);  // draw the circle
		std::sprintf(acPtIdx, "%d", i);
		cv::putText(oImgPlt, acPtIdx, m_vo2dPt[i], cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
	}

	// plot the projected 2D points
		// plot the 2D points
	for (int i = 0; i < m_vo2dPt.size(); i++)
	{
		cv::Mat o3dPtMat(3, 1, CV_64F);
		cv::Mat o2dPtMat(3, 1, CV_64F);

		o3dPtMat.at<double>(0, 0) = m_vo3dPt[i].x;
		o3dPtMat.at<double>(1, 0) = m_vo3dPt[i].y;
		o3dPtMat.at<double>(2, 0) = 1;
		o2dPtMat = m_oHomoMat * o3dPtMat;

		cv::circle(oImgPlt, cv::Point2f((o2dPtMat.at<double>(0, 0) / o2dPtMat.at<double>(2, 0)), (o2dPtMat.at<double>(1, 0) / o2dPtMat.at<double>(2, 0))),
			12, cv::Scalar(0, 0, 255), 1, LINE_AA);  // draw the circle
	}

	// display plotted image
	cv::namedWindow("3D grid on the ground plane", WINDOW_NORMAL);
	cv::imshow("3D grid on the ground plane", oImgPlt);
	cv::waitKey(0);
	cv::destroyAllWindows();

	// save plotted image
	if (m_oCfg.getOutCalDispFlg())
		cv::imwrite(m_oCfg.getOutCalDispPth(), oImgPlt);
}


C2dPtSel::C2dPtSel(void)
{

}

C2dPtSel::~C2dPtSel(void)
{

}

void C2dPtSel::initialize(CCfg oCfg, cv::Mat oImgFrm)
{
	// configuration parameters
	m_oCfg = oCfg;

	// frame image for plotting results
	m_oImgFrm = oImgFrm.clone();
}

std::vector<cv::Point> C2dPtSel::process(void)
{
	std::vector<cv::Point> voVanPt;

	if (m_oCfg.getCalSel2dPtFlg())
	{
		std::cout << "Hot keys: \n"
			<< "\tESC - exit\n"
			<< "\tr - re-select a set of 2D points\n"
			<< "\to - finish selecting a set of 2D points\n"
			<< std::endl;

		cv::Mat oImgFrm = m_oImgFrm.clone();

		cv::namedWindow("selector of 2D points", WINDOW_NORMAL);
		cv::imshow("selector of 2D points", m_oImgFrm);
		cv::setMouseCallback("selector of 2D points", on_mouse);  // register for mouse event

		while (1)
		{
			int nKey = cv::waitKey(0);	// read keyboard event

			if (nKey == 27)
				break;

			if (nKey == 'r')  // reset the nodes
			{
				std::vector<cv::Point>().swap(m_voNd);
				m_oImgFrm = oImgFrm.clone();
				cv::imshow("selector of 2D points", m_oImgFrm);
			}

			if (nKey == 'o')	// finish selection of pairs of test points
			{
				cv::destroyWindow("selector of 2D points");
				std::vector<cv::Point2f> vo3dPt = m_oCfg.getCal3dPtLs();
				if (vo3dPt.size() == m_voNd.size())
					return m_voNd;
				else
				{
					std::cout << "Error: Mis-match between the number of selected 2D points and the number of 3D points." << std::endl;
					break;
				}
			}
		}
	}
}

void C2dPtSel::addNd(int nX, int nY)
{
	char acNdIdx[32];
	cv::Point oCurrNd;
	oCurrNd.x = nX;
	oCurrNd.y = nY;

	m_voNd.push_back(oCurrNd);
	// std::cout << "current node(" << oCurrNd.x << "," << oCurrNd.y << ")" << std::endl;	// for debug
	cv::circle(m_oImgFrm, oCurrNd, 6, cv::Scalar(255, 0, 0), 1, LINE_AA);  // draw the circle
	std::sprintf(acNdIdx, "%d", (int)(m_voNd.size() - 1));
	cv::putText(m_oImgFrm, acNdIdx, oCurrNd, cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
	cv::imshow("selector of 2D points", m_oImgFrm);
}
