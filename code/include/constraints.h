#ifndef CONSTRAINTS_HEADER_FILE
#define CONSTRAINTS_HEADER_FILE

using namespace Eigen;
using namespace std;

typedef enum ConstraintType{DISTANCE, COLLISION} ConstraintType;   //You can expand it for more constraints
typedef enum ConstraintEqualityType{EQUALITY, INEQUALITY} ConstraintEqualityType;

//there is such constraints per two variables that are equal. That is, for every attached vertex there are three such constraints for (x,y,z);
class Constraint{
public:
  
  int m1, m2;                     //Two participating meshes (can be the same)  - auxiliary data for users (constraint class shouldn't use that)
  int v1, v2;                     //Two vertices from the respective meshes - auxiliary data for users (constraint class shouldn't use that)
  double invMass1, invMass2;       //inverse masses of two bodies
  double refValue;                //Reference values to use in the constraint, when needed (like distance)
  bool isUpper;                   //in case this is an inequality constraints, whether it's an upper or a lower bound
  RowVector3d refVector;             //Reference vector when needed (like vector)
  double CRCoeff;                 //velocity bias
  ConstraintType constraintType;  //The type of the constraint, and will affect the value and the gradient. This SHOULD NOT change after initialization!
  ConstraintEqualityType constraintEqualityType;  //whether the constraint is an equality or an inequality
  
  Constraint(const ConstraintType _constraintType, const ConstraintEqualityType _constraintEqualityType, const bool _isUpper, const int& _m1, const int& _v1, const int& _m2, const int& _v2, const double& _invMass1, const double& _invMass2, const RowVector3d& _refVector, const double& _refValue, const double& _CRCoeff):constraintType(_constraintType), constraintEqualityType(_constraintEqualityType), isUpper(_isUpper), m1(_m1), v1(_v1), m2(_m2), v2(_v2), invMass1(_invMass1), invMass2(_invMass2),  refValue(_refValue), CRCoeff(_CRCoeff){
    refVector=_refVector;
  }
  
  ~Constraint(){}
  
  
  
  //computes the impulse needed for all particles to resolve the velocity constraint, and corrects the velocities accordingly.
  //The velocities are a vector (vCOM1, w1, vCOM2, w2) in both input and output.
  //returns true if constraint was already valid with "currVelocities", and false otherwise (false means there was a correction done)
  bool resolve_velocity_constraint(const MatrixXd& currCOMPositions, const MatrixXd& currVertexPositions, const MatrixXd& currCOMVelocities, const MatrixXd& currAngVelocities, const Matrix3d& invInertiaTensor1, const Matrix3d& invInertiaTensor2, MatrixXd& correctedCOMVelocities, MatrixXd& correctedAngVelocities, double tolerance) {
    // We first obtain the vector between the COMs and the contact points
    Vector3d rA = (currVertexPositions.row(v1) - currCOMPositions.row(0)).transpose();
    Vector3d rB = (currVertexPositions.row(v2) - currCOMPositions.row(1)).transpose();

    // We then obtain the normal vector of the constraint (slide 21 of lecture 9)
    Vector3d n = (currVertexPositions.row(v1) - currVertexPositions.row(v2)).normalized().transpose();
    
    // Current velocities (v_A+ and v_B+ in the formula)
    Vector3d vA = currCOMVelocities.row(0).transpose();
    Vector3d wA = currAngVelocities.row(0).transpose();
    Vector3d vB = currCOMVelocities.row(1).transpose();
    Vector3d wB = currAngVelocities.row(1).transpose();

    // Relative velocity at the contact point (Lecture 3 slide 15)
    Vector3d combinedVelA = vA + wA.cross(rA);
    Vector3d combinedVelB = vB + wB.cross(rB);
    Vector3d vrel = combinedVelA - combinedVelB;
    double normalVel = vrel.dot(n);

    // Check if constraint is already satisfied (normal velocity is close to 0)
    // i.e. the objects are not moving relative to each other
    if (abs(normalVel) < tolerance) {
        correctedCOMVelocities = currCOMVelocities;
        correctedAngVelocities = currAngVelocities;
        return true;
    }

    // J matrix
    MatrixXd J(1, 12);
    J << n.transpose(),                  
         (rA.cross(n)).transpose(),      
         -n.transpose(),                 
         -(rB.cross(n)).transpose();     

    // M inverse matrix
    MatrixXd Minv = MatrixXd::Zero(12, 12);
    Minv.block<3,3>(0,0) = Matrix3d::Identity() * invMass1;
    Minv.block<3,3>(3,3) = invInertiaTensor1;
    Minv.block<3,3>(6,6) = Matrix3d::Identity() * invMass2;
    Minv.block<3,3>(9,9) = invInertiaTensor2;

    // Calculate lambda
    double lambda = -(1.0 + CRCoeff) * normalVel / (J * Minv * J.transpose())(0,0);

    // We calculate the impulse vector (j * n)
    Vector3d impulse = lambda * n;

    // We calculate the torques (r x F)
    Vector3d torque1 = rA.cross(impulse);
    Vector3d torque2 = rB.cross(impulse);
    
    // Apply corrections
    correctedCOMVelocities = currCOMVelocities;
    correctedAngVelocities = currAngVelocities;

    // Linear velocity corrections
    correctedCOMVelocities.row(0) += impulse.transpose() * invMass1;
    correctedCOMVelocities.row(1) -= impulse.transpose() * invMass2;

    // Angular velocity corrections (lighter object should rotate more as well)
    correctedAngVelocities.row(0) += (invInertiaTensor1 * torque1).transpose();
    correctedAngVelocities.row(1) -= (invInertiaTensor2 * torque2).transpose();

    return false;
  }
  
  //projects the position unto the constraint
  //returns true if constraint was already good
  bool resolve_position_constraint(const MatrixXd& currCOMPositions, const MatrixXd& currConstPositions, MatrixXd& correctedCOMPositions, double tolerance) {
    // Get vector between constrained points
    RowVector3d constraintVector = currConstPositions.row(1) - currConstPositions.row(0);
    double currentDistance = constraintVector.norm();
    
    // Get normalized direction (Jacobian)
    // - The sign changes depending on the nature of the constraint
    RowVector3d direction = constraintVector.normalized();

    // We give a value by default and then we will correct it
    correctedCOMPositions = currCOMPositions;

    if (isUpper) {
        if (currentDistance <= refValue + tolerance) {
            return true;
        }
        double correction = currentDistance - refValue;
        double totalInvMass = invMass1 + invMass2;
        // If at least one object is movable
        if (totalInvMass > 0) {  
            // The lighter object moves more, so we scale by the inverse of the mass
            // Formula taken from lecture 9 slide 21
            correctedCOMPositions.row(0) += (correction * invMass1/totalInvMass) * direction;
            correctedCOMPositions.row(1) -= (correction * invMass2/totalInvMass) * direction;
        }
        return false;  // We made a correction
    } else {
        if (currentDistance >= refValue - tolerance) {
            return true;
        }
        double correction = refValue - currentDistance;
        double totalInvMass = invMass1 + invMass2;
        if (totalInvMass > 0) {
            correctedCOMPositions.row(0) -= (correction * invMass1/totalInvMass) * direction;
            correctedCOMPositions.row(1) += (correction * invMass2/totalInvMass) * direction;
        }
        return false;
    }
  }
  
};



#endif /* constraints_h */
