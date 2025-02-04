#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#include <polyscope/curve_network.h>
#include "readOFF.h"
#include "scene.h"
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <set>
#include <array>
#include <chrono>

using namespace Eigen;
using namespace std;

bool isAnimating = false;

polyscope::SurfaceMesh* pMesh;
polyscope::CurveNetwork* pConstraints;

double currTime = 0;
double timeStep = 0.02; //assuming 50 fps
double CRCoeff = 1.0;
double tolerance = 1e-3;
int maxIterations = 10000;


Scene scene;

void callback_function() {
  ImGui::PushItemWidth(50);
  
  ImGui::TextUnformatted("Animation Parameters");
  ImGui::Separator();
  bool changed = ImGui::Checkbox("isAnimating", &isAnimating);
  ImGui::PopItemWidth();
  if (!isAnimating)
    return;
  
  scene.update_scene(timeStep, CRCoeff, maxIterations, tolerance);
  
  pMesh->updateVertexPositions(scene.currV);
  pConstraints->updateNodePositions(scene.currConstVertices);
}


int main()
{
  
  scene.load_scene("two_cylinder-scene.txt","no-constraints.txt");
  polyscope::init();
  
  scene.update_scene(0.0, CRCoeff, maxIterations, tolerance);
  
  // Visualization
  pMesh = polyscope::registerSurfaceMesh("Entire Scene", scene.currV, scene.allF);
  pConstraints = polyscope::registerCurveNetwork("Constraints", scene.currConstVertices, scene.constEdges);
  polyscope::options::groundPlaneHeightMode = polyscope::GroundPlaneHeightMode::Manual;
  polyscope::options::groundPlaneHeight = 0.; // in world coordinates along the up axis
  polyscope::state::userCallback = callback_function;
  
  polyscope::show();
  
}

