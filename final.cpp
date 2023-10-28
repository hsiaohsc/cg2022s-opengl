#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;

/***freeglut***/
#include ".\gl\freeglut.h"

#define PI 3.14159265
#define NUM_SPHERES 30
#define GL_MULTISAMPLE_ARB 0x809D
#define GL_CLAMP_TO_EDGE 0x812F

// Vertex structure
typedef struct vertex {
    // coordinate
    float x;
    float y;
    float z;
} vertex;

// Face structure
typedef struct face {
    // vertex ids
    int v0;
    int v1;
    int v2;
} face;

// Color storage
typedef struct RGBColor
{
    float r;
    float g;
    float b;
};

int curr_obj_id = -1;
string objFiles[] = { "D:\\CG_final\\gourd.obj", "D:\\CG_final\\octahedron.obj", "D:\\CG_final\\teapot.obj", "D:\\CG_final\\teddy.obj", 
                      "D:\\CG_final\\dolphin.obj", "D:\\CG_final\\starfish_v2.obj" };
RGBColor mainColor = { 1.0f,1.0f,1.0f };
vertex objmax, objmin;
bool sw_boundingbox = true;

int rasterizeMode = 2;  //0: point, 1:line, 2:face

vector <vertex> main_vertices, star_vertices, dolphin_vertices;
vector <face> main_faces, star_faces, dolphin_faces;
int num_main_v = 0, num_main_f = 0;
int num_star_v = 0, num_dolphin_v = 0, num_star_f = 0, num_dolphin_f = 0;

void normCrossProd(float u[3], float v[3], float out[3]);
void buildPopupMenu();
bool readFile(string objpath, vector<vertex>& v, vector<face>& f, int& nv, int& nf, float objscale);
void delay(float secs);


// Light and material Data
GLfloat fLightPos[4] = { -100.0f, 100.0f, 50.0f, 1.0f };  // Point source
GLfloat fNoLight[] = { 0.0f, 0.0f, 0.0f, 0.0f };
GLfloat fLowLight[] = { 0.25f, 0.25f, 0.25f, 1.0f };
GLfloat fBrightLight[] = { 1.0f, 1.0f, 1.0f, 1.0f };

// Transformation matrix to project shadow
float mShadowMatrix[16];    // 4 X 4

#define SEA_TEXTURE  0
#define SAND_TEXTURE  3
#define ROBOT_TEXTURE   1
#define DOLPHIN_TEXTURE 2
#define STARFISH_TEXTURE 4
#define NUM_TEXTURES    5
GLuint  textureObjects[NUM_TEXTURES];

const char* szTextureFiles[] = { "D:\\CG_final\\water.jpg", "D:\\CG_final\\magic.jpg", "D:\\CG_final\\star2.jpg", "D:\\CG_final\\sand.jpg", "D:\\CG_final\\starfish2.jpg" };

int timer_cnt = 0;
int timer_flag = 1;


void normalize(float v[3]) {    // normalize vector
    GLfloat d = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    v[0] /= d;
    v[1] /= d;
    v[2] /= d;
}

void normCrossProd(float u[3], float v[3], float out[3]) {  // cross product
    out[0] = u[1] * v[2] - u[2] * v[1];
    out[1] = u[2] * v[0] - u[0] * v[2];
    out[2] = u[0] * v[1] - u[1] * v[0];
    normalize(out);
}


/////////////////////////////////////////////////////////////
// For best results, put this in a display list
// Draw a sphere at the origin
void gltDrawSphere(GLfloat fRadius, GLint iSlices, GLint iStacks)
{
    GLfloat drho = (GLfloat)(3.141592653589) / (GLfloat)iStacks;
    GLfloat dtheta = 2.0f * (GLfloat)(3.141592653589) / (GLfloat)iSlices;
    GLfloat ds = 1.0f / (GLfloat)iSlices;
    GLfloat dt = 1.0f / (GLfloat)iStacks;
    GLfloat t = 1.0f;
    GLfloat s = 0.0f;
    GLint i, j;     // Looping variables

    for (i = 0; i < iStacks; i++)
    {
        GLfloat rho = (GLfloat)i * drho;
        GLfloat srho = (GLfloat)(sin(rho));
        GLfloat crho = (GLfloat)(cos(rho));
        GLfloat srhodrho = (GLfloat)(sin(rho + drho));
        GLfloat crhodrho = (GLfloat)(cos(rho + drho));

        // Many sources of OpenGL sphere drawing code uses a triangle fan
        // for the caps of the sphere. This however introduces texturing 
        // artifacts at the poles on some OpenGL implementations
        glBegin(GL_TRIANGLE_STRIP);
        s = 0.0f;
        for (j = 0; j <= iSlices; j++)
        {
            GLfloat theta = (j == iSlices) ? 0.0f : j * dtheta;
            GLfloat stheta = (GLfloat)(-sin(theta));
            GLfloat ctheta = (GLfloat)(cos(theta));

            GLfloat x = stheta * srho;
            GLfloat y = ctheta * srho;
            GLfloat z = crho;

            glTexCoord2f(s, t);
            glNormal3f(x, y, z);
            glVertex3f(x * fRadius, y * fRadius, z * fRadius);

            x = stheta * srhodrho;
            y = ctheta * srhodrho;
            z = crhodrho;
            glTexCoord2f(s, t - dt);
            s += ds;
            glNormal3f(x, y, z);
            glVertex3f(x * fRadius, y * fRadius, z * fRadius);
        }
        glEnd();

        t -= dt;
    }
}


void GetPlaneNormal(float n[3] , vertex p1, vertex p2, vertex p3) {
    float v1[3], v2[3];

    // V1 = p3 - p1
    v1[0] = p3.x - p1.x;
    v1[1] = p3.y - p1.y;
    v1[2] = p3.z - p1.z;

    // V2 = P2 - p1
    v2[0] = p2.x - p1.x;
    v2[1] = p2.y - p1.y;
    v2[2] = p2.z - p1.z;

    // Unit normal to plane - Not sure which is the best way here
    // Cross Product and Normalize Vector
    normCrossProd(v1, v2, n);
}

void GetPlaneEquation(float planeEq[4], const float p1[3], const float p2[3], const float p3[3]) {
    // Get two vectors... do the cross product
    float v1[3], v2[3];

    // V1 = p3 - p1
    v1[0] = p3[0] - p1[0];
    v1[1] = p3[1] - p1[1];
    v1[2] = p3[2] - p1[2];

    // V2 = P2 - p1
    v2[0] = p2[0] - p1[0];
    v2[1] = p2[1] - p1[1];
    v2[2] = p2[2] - p1[2];

    // Unit normal to plane - Not sure which is the best way here
    // Cross Product and Normalize Vector
    normCrossProd(v1, v2, planeEq);

    // Back substitute to get D
    planeEq[3] = -(planeEq[0] * p3[0] + planeEq[1] * p3[1] + planeEq[2] * p3[2]);
}

void MakePlanarShadowMatrix(float proj[16], float planeEq[4], float vLightPos[3]) {
    // These just make the code below easier to read. They will be 
    // removed by the optimizer.	
    float a = planeEq[0];
    float b = planeEq[1];
    float c = planeEq[2];
    float d = planeEq[3];

    float dx = -vLightPos[0];
    float dy = -vLightPos[1];
    float dz = -vLightPos[2];

    // Now build the projection matrix
    proj[0] = b * dy + c * dz;
    proj[1] = -a * dy;
    proj[2] = -a * dz;
    proj[3] = 0.0;

    proj[4] = -b * dx;
    proj[5] = a * dx + c * dz;
    proj[6] = -b * dz;
    proj[7] = 0.0;

    proj[8] = -c * dx;
    proj[9] = -c * dy;
    proj[10] = a * dx + b * dy;
    proj[11] = 0.0;

    proj[12] = -d * dx;
    proj[13] = -d * dy;
    proj[14] = -d * dz;
    proj[15] = a * dx + b * dy + c * dz;
    // Shadow matrix ready
}

//////////////////////////////////////////////////////////////////
// This function does any needed initialization on the rendering
// context. 
void SetupRC()
{
    GLfloat vPoints[3][3] = { { 0.0f, -0.4f, 0.0f },
                             { 10.0f, -0.4f, 0.0f },
                             { 5.0f, -0.4f, -5.0f } };
    int i;
    

    // Grayish background
    //glClearColor(fLowLight[0], fLowLight[1], fLowLight[2], fLowLight[3]);
    glClearColor(0.5, 0.8, 1.0, fLowLight[3]);

    // Clear stencil buffer with zero, increment by one whenever anybody
    // draws into it. When stencil function is enabled, only write where
    // stencil value is zero. This prevents the transparent shadow from drawing
    // over itself
    glStencilOp(GL_INCR, GL_INCR, GL_INCR);
    glClearStencil(0);
    glStencilFunc(GL_EQUAL, 0x0, 0x01);

    // Cull backs of polygons
    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);
    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE_ARB);

    // Setup light parameters
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, fNoLight);
    glLightfv(GL_LIGHT0, GL_AMBIENT, fLowLight);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, fBrightLight);
    glLightfv(GL_LIGHT0, GL_SPECULAR, fBrightLight);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);

    // Calculate shadow matrix
    GLfloat pPlane[4];
    GetPlaneEquation(pPlane, vPoints[0], vPoints[1], vPoints[2]);
    MakePlanarShadowMatrix(mShadowMatrix, pPlane, fLightPos);

    // Mostly use material tracking
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
    glMaterialfv(GL_FRONT, GL_SPECULAR, fBrightLight);
    glMateriali(GL_FRONT, GL_SHININESS, 128);


    // Set up texture maps
    glEnable(GL_TEXTURE_2D);
    glGenTextures(NUM_TEXTURES, textureObjects);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);


    for (i = 0; i < NUM_TEXTURES; i++)
    {

        Mat image1 = imread(szTextureFiles[i]);
        if (image1.empty()) {
            cout << "image" << i << " empty" << endl;
        }
        else {
            glBindTexture(GL_TEXTURE_2D, textureObjects[i]);
            //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image1.cols, image1.rows, 0,
                GL_BGR_EXT, GL_UNSIGNED_BYTE, image1.ptr());
        }
    }

}


////////////////////////////////////////////////////////////////////////
// Do shutdown for the rendering context
void ShutdownRC(void)
{
    // Delete the textures
    glDeleteTextures(NUM_TEXTURES, textureObjects);
}

///////////////////////////////////////////////////////////
// Draw the ground as a series of triangle strips
void DrawGround(int texture, float l, float h)
{
    //GLfloat fExtent = 20.0f;
    GLfloat fExtent = l;
    GLfloat hExtent = h;
    GLfloat fStep = 1.0f;
    GLfloat y = -0.4f;
    GLfloat iStrip, iRun;
    GLfloat s = 0.0f;
    GLfloat t = 0.0f;
    //GLfloat texStep = 1.0f / (fExtent * .075f);
    GLfloat texStep = 1.0f / (20 * .075f);

    glBindTexture(GL_TEXTURE_2D, textureObjects[texture]);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    for (iStrip = -fExtent; iStrip <= fExtent; iStrip += fStep)
    {
        t = 0.0f;
        glBegin(GL_TRIANGLE_STRIP);

        //for (iRun = fExtent; iRun >= -fExtent; iRun -= fStep)
        for (iRun = hExtent; iRun >= -hExtent; iRun -= fStep)
        {
            glTexCoord2f(s, t);
            glNormal3f(0.0f, 1.0f, 0.0f);   // All Point up
            glVertex3f(iStrip, y, iRun);

            glTexCoord2f(s + texStep, t);
            glNormal3f(0.0f, 1.0f, 0.0f);   // All Point up
            glVertex3f(iStrip + fStep, y, iRun);

            t += texStep;
        }
        glEnd();
        s += texStep;
    }
}


// Draw rectangle cubes
void drawRectangle(float l, float h) {
    if (h < 0 || l < 0) return;
    float x = -l / 2;
    float y = -h / 2;
    float z = -l / 2;

    glBindTexture(GL_TEXTURE_2D, textureObjects[ROBOT_TEXTURE]);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    
    // Front face (outside)
    glBegin(GL_POLYGON); // draw a textured quad 
    glNormal3f(0.0f, 0.0f, 1.0f);   // All Point front
    glTexCoord2f(0.0, 0.0); glVertex3f(x, y, z + l);
    glTexCoord2f(0.0, 1.0); glVertex3f(x + l, y, z + l);
    glTexCoord2f(1.0, 1.0); glVertex3f(x + l, y + h, z + l);
    glTexCoord2f(1.0, 0.0); glVertex3f(x, y + h, z + l);
    glEnd();
    // Front face (inside)
    glBegin(GL_POLYGON); // draw a textured quad 
    glTexCoord2f(1.0, 0.0); glVertex3f(x, y + h, z + l);
    glTexCoord2f(1.0, 1.0); glVertex3f(x + l, y + h, z + l);
    glTexCoord2f(0.0, 1.0); glVertex3f(x + l, y, z + l);
    glTexCoord2f(0.0, 0.0); glVertex3f(x, y, z + l);
    glEnd();

    // Back face (outside)
    glBegin(GL_POLYGON); // draw a textured quad 
    glNormal3f(0.0f, 0.0f, -1.0f);   // All Point back
    glTexCoord2f(0.0, 0.0); glVertex3f(x, y, z);
    glTexCoord2f(0.0, 1.0); glVertex3f(x + l, y, z);
    glTexCoord2f(1.0, 1.0); glVertex3f(x + l, y + h, z);
    glTexCoord2f(1.0, 0.0); glVertex3f(x, y + h, z);
    glEnd();
    // Back face (inside)
    glBegin(GL_POLYGON); // draw a textured quad 
    glTexCoord2f(1.0, 0.0); glVertex3f(x, y + h, z);
    glTexCoord2f(1.0, 1.0); glVertex3f(x + l, y + h, z);
    glTexCoord2f(0.0, 1.0); glVertex3f(x + l, y, z);
    glTexCoord2f(0.0, 0.0); glVertex3f(x, y, z);
    glEnd();

    // Left face (outside)
    glBegin(GL_POLYGON); // draw a textured quad 
    glNormal3f(-1.0f, 0.0f, 0.0f);   // All Point left
    glTexCoord2f(0.0, 0.0); glVertex3f(x, y, z);
    glTexCoord2f(0.0, 1.0); glVertex3f(x, y, z + l);
    glTexCoord2f(1.0, 1.0); glVertex3f(x, y + h, z + l);
    glTexCoord2f(1.0, 0.0); glVertex3f(x, y + h, z);
    glEnd();
    // Left face (inside)
    glBegin(GL_POLYGON); // draw a textured quad 
    glTexCoord2f(1.0, 0.0); glVertex3f(x, y + h, z);
    glTexCoord2f(1.0, 1.0); glVertex3f(x, y + h, z + l);
    glTexCoord2f(0.0, 1.0); glVertex3f(x, y, z + l);
    glTexCoord2f(0.0, 0.0); glVertex3f(x, y, z);
    glEnd();

    // Right face (outside)
    glBegin(GL_POLYGON); // draw a textured quad 
    glNormal3f(1.0f, 0.0f, 0.0f);   // All Point right
    glTexCoord2f(0.0, 0.0); glVertex3f(x + l, y, z);
    glTexCoord2f(0.0, 1.0); glVertex3f(x + l, y, z + l);
    glTexCoord2f(1.0, 1.0); glVertex3f(x + l, y + h, z + l);
    glTexCoord2f(1.0, 0.0); glVertex3f(x + l, y + h, z);
    glEnd();
    // Right face (inside)
    glBegin(GL_POLYGON); // draw a textured quad 
    glTexCoord2f(1.0, 0.0); glVertex3f(x + l, y + h, z);
    glTexCoord2f(1.0, 1.0); glVertex3f(x + l, y + h, z + l);
    glTexCoord2f(0.0, 1.0); glVertex3f(x + l, y, z + l);
    glTexCoord2f(0.0, 0.0); glVertex3f(x + l, y, z);
    glEnd();

    // Down face (outside)
    glBegin(GL_POLYGON); // draw a textured quad 
    glNormal3f(0.0f, -1.0f, 0.0f);   // All Point down
    glTexCoord2f(0.0, 0.0); glVertex3f(x, y, z);
    glTexCoord2f(0.0, 1.0); glVertex3f(x + l, y, z);
    glTexCoord2f(1.0, 1.0); glVertex3f(x + l, y, z + l);
    glTexCoord2f(1.0, 0.0); glVertex3f(x, y, z + l);
    glEnd();
    // Down face (inside)
    glBegin(GL_POLYGON); // draw a textured quad 
    glTexCoord2f(1.0, 0.0); glVertex3f(x, y, z + l);
    glTexCoord2f(1.0, 1.0); glVertex3f(x + l, y, z + l);
    glTexCoord2f(0.0, 1.0); glVertex3f(x + l, y, z);
    glTexCoord2f(0.0, 0.0); glVertex3f(x, y, z);
    glEnd();

    // Up face (outside)
    glBegin(GL_POLYGON); // draw a textured quad 
    glNormal3f(0.0f, 1.0f, 0.0f);   // All Point up
    glTexCoord2f(0.0, 0.0); glVertex3f(x, y + h, z);
    glTexCoord2f(0.0, 1.0); glVertex3f(x + l, y + h, z);
    glTexCoord2f(1.0, 1.0); glVertex3f(x + l, y + h, z + l);
    glTexCoord2f(1.0, 0.0); glVertex3f(x, y + h, z + l);
    glEnd();
    // Up face (inside)
    glBegin(GL_POLYGON); // draw a textured quad 
    glTexCoord2f(1.0, 0.0); glVertex3f(x, y + h, z + l);
    glTexCoord2f(1.0, 1.0); glVertex3f(x + l, y + h, z + l);
    glTexCoord2f(0.0, 1.0); glVertex3f(x + l, y + h, z);
    glTexCoord2f(0.0, 0.0); glVertex3f(x, y + h, z);

    glEnd();

}

void drawObj(vector<vertex> &v, vector<face> &f, int num_f, int texture) {
    GLfloat n[3];

    if (texture >= 0) {
        glBindTexture(GL_TEXTURE_2D, textureObjects[texture]);
    }

    for (int i = 0; i < num_f; i++) {
        GetPlaneNormal(n, v[f[i].v2], v[f[i].v1], v[f[i].v0]);
        //GetPlaneEquation(n[4], const float p1[3], const float p2[3], const float p3[3]);

        // Front face (outside)
        glBegin(GL_POLYGON); // draw a textured quad 
        glNormal3f(n[0], n[1], n[2]);
        glTexCoord2f(0.0, 0.0); glVertex3f(v[f[i].v0].x, v[f[i].v0].y, v[f[i].v0].z);
        glTexCoord2f(0.0, 1.0); glVertex3f(v[f[i].v1].x, v[f[i].v1].y, v[f[i].v1].z);
        glTexCoord2f(1.0, 1.0); glVertex3f(v[f[i].v2].x, v[f[i].v2].y, v[f[i].v2].z);
        glEnd();

        // Back face (inside)
        glBegin(GL_POLYGON); // draw a textured quad 
        glNormal3f(-n[0], -n[1], -n[2]);
        glTexCoord2f(1.0, 1.0); glVertex3f(v[f[i].v2].x, v[f[i].v2].y, v[f[i].v2].z);
        glTexCoord2f(0.0, 1.0); glVertex3f(v[f[i].v1].x, v[f[i].v1].y, v[f[i].v1].z);
        glTexCoord2f(0.0, 0.0); glVertex3f(v[f[i].v0].x, v[f[i].v0].y, v[f[i].v0].z);
        glEnd();

    }
    

}

void drawBoundingBox() {
    glBegin(GL_LINE_STRIP);
    glVertex3f(objmax.x, objmax.y, objmax.z);
    glVertex3f(objmin.x, objmax.y, objmax.z);
    glVertex3f(objmin.x, objmin.y, objmax.z);
    glVertex3f(objmax.x, objmin.y, objmax.z);
    glVertex3f(objmax.x, objmax.y, objmax.z);
    glEnd();
    glBegin(GL_LINE_STRIP);
    glVertex3f(objmax.x, objmin.y, objmin.z);
    glVertex3f(objmin.x, objmin.y, objmin.z);
    glVertex3f(objmin.x, objmax.y, objmin.z);
    glVertex3f(objmax.x, objmax.y, objmin.z);
    glVertex3f(objmax.x, objmin.y, objmin.z);
    glEnd();
    glBegin(GL_LINES);
    glVertex3f(objmax.x, objmax.y, objmax.z);
    glVertex3f(objmax.x, objmax.y, objmin.z);
    glVertex3f(objmin.x, objmax.y, objmax.z);
    glVertex3f(objmin.x, objmax.y, objmin.z);
    glVertex3f(objmax.x, objmin.y, objmax.z);
    glVertex3f(objmax.x, objmin.y, objmin.z);
    glVertex3f(objmin.x, objmin.y, objmax.z);
    glVertex3f(objmin.x, objmin.y, objmin.z);
    glEnd();
}

void drawRobot(float legRot, float handRot) {
    // Draw torso
    drawRectangle(0.2f, 0.3f);
    // Draw head
    glPushMatrix();
    glTranslatef(0.0f, 0.25f, 0.0f);
    drawRectangle(0.1f, 0.1f);
    glPopMatrix();
    // Draw left leg
    glPushMatrix();
    glTranslatef(0.06f, -0.25f, 0.0f);
    glRotatef(legRot, 0.0f, 0.0f, 1.0f);
    drawRectangle(0.08f, 0.15f);
    glPushMatrix();
    glTranslatef(0.0f, -0.2f, 0.0f);
    glRotatef(-legRot * 1.2, 1.0f, 0.0f, 0.0f);
    drawRectangle(0.08f, 0.15f);
    glPopMatrix();
    glPopMatrix();
    // Draw right leg
    glPushMatrix();
    glTranslatef(-0.06f, -0.25f, 0.0f);
    glRotatef(-legRot, 0.0f, 0.0f, 1.0f);
    drawRectangle(0.08f, 0.15f);
    glPushMatrix();
    glTranslatef(0.0f, -0.2f, 0.0f);
    glRotatef(legRot * 2, 1.0f, 0.0f, 0.0f);
    drawRectangle(0.08f, 0.15f);
    glPopMatrix();
    glPopMatrix();
    // Draw left arm
    glPushMatrix();
    glTranslatef(0.15f, 0.07f, 0.0f);
    glRotatef(-handRot * 6, 1.0f, 0.0f, 0.0f);
    glRotatef(handRot, 0.0f, 0.0f, 1.0f);
    drawRectangle(0.06f, 0.15f);
    glPushMatrix();
    glTranslatef(0.0f, -0.18f, 0.0f);
    glRotatef(-handRot, 1.0f, 0.0f, -1.0f);
    drawRectangle(0.06f, 0.15f);
    glPopMatrix();
    glPopMatrix();
    // Draw right arm
    glPushMatrix();
    glTranslatef(-0.15f, 0.07f, 0.0f);
    glRotatef(-handRot * 6, 1.0f, 0.0f, 0.0f);
    glRotatef(-handRot, 0.0f, 0.0f, 1.0f);
    drawRectangle(0.06f, 0.15f);
    glPushMatrix();
    glTranslatef(0.0f, -0.18f, 0.0f);
    glRotatef(handRot, 1.0f, 0.0f, -1.0f);
    drawRectangle(0.06f, 0.15f);
    glPopMatrix();
    glPopMatrix();
}

///////////////////////////////////////////////////////////////////////
// Draw random inhabitants and the rotating torus/sphere duo
void DrawInhabitants(GLint nShadow)
{
    static GLfloat yRot = 0.0f;         // Rotation angle for animation
    static GLfloat handRot = 0.0f;     // Rotation angle for hand animation
    static GLfloat hand_toadd = 0.5f;     // Rotation angle hand animation
    static GLfloat legRot = 0.0f;     // Rotation angle for leg animation
    static GLfloat leg_toadd = +0.5f;     // Rotation angle for leg animation
    static GLfloat jump = -0.48f;     // jump height for robot animation
    static GLfloat jump_toadd = 0.003f;     // jump height for robot animation
    static GLfloat starMove = 0.0f;         // Translate x for starfish animation
    static GLfloat starM_toadd = 0.001f;         // Translate x for starfish animation
    static GLfloat dolRot = -30.0f;         // Rotation angle for dolphin animation
    static GLfloat dolRot_toadd = 0.2f;         // Rotation angle for dolphin animation
    static GLint dolFace = 0;         // Face angle for dolphin animation
    static GLfloat dolMove = -0.5f;         // Translate x for dolphin animation
    static GLfloat dolM_toadd = 0.003f;         // Translate x for dolphin animation
    static GLfloat dolJump = 0.0f;         // Translate y for dolphin animation
    static GLfloat dolJ_toadd = 0.003f;         // Translate y for dolphin animation
    

    if (nShadow == 0)
    {
        // Robot animation
        yRot += 0.5f;
        if (yRot > 360) yRot -= 360;

        if (legRot > 30 || legRot < 0) {   // hands and legs swing between 0~30 degrees
            hand_toadd *= -1;
            leg_toadd *= -1;
            jump_toadd *= -1;
        }
        handRot += hand_toadd;
        legRot += leg_toadd;
        jump += jump_toadd;

        // Starfish anumation
        
        if (starMove > 0.5 || starMove < -0.5) {
            starM_toadd *= -1;
        }
        starMove += starM_toadd;

        // Dolphin animation
        if (dolRot > 40 || dolRot<-40) {
            dolRot_toadd *= -1;
            //dolRot = -30;
        }

        if (dolRot >= 0 && dolRot < 0.1) {
            dolJ_toadd *= -1;
        }
        if (dolMove > 3.5 || dolMove < -1.5) {
            dolM_toadd *= -1;

            dolFace += 180;
            if (dolFace >= 360) {
                dolFace -= 360;
            }
        }
        dolRot += dolRot_toadd;
        dolMove += dolM_toadd;
        dolJump += dolJ_toadd;
        //cout << dolRot <<"\t"<<dolJump << endl;
        
        
        glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
        
    }
    else {
        glColor4f(0.00f, 0.00f, 0.00f, .6f);  // Shadow color
        //glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
    


    glPushMatrix(); //0
    glTranslatef(0.0f, 0.1f, -2.5f);

    // ============ Draw Robot ============
    glPushMatrix(); //3
    glScalef(0.5f, 0.5f, 0.5f);
    glTranslatef(-1.5f, jump, 0.7f);
    glRotatef(yRot, 0.0f, 1.0f, 0.0f);
    drawRobot(legRot, handRot);
    glPopMatrix();  //3

    glEnable(GL_TEXTURE_2D);
    // ============ Draw Starfish ============
    glPushMatrix(); //4
    glTranslatef(starMove, -0.485f, 0.98f);
    glRotatef(yRot, 0.0f, 1.0f, 0.0f);
    glRotatef(-90, 1.0f, 0.0f, 0.0f);
    drawObj(star_vertices, star_faces, num_star_f, STARFISH_TEXTURE);
    glPopMatrix();  //4
    
    // ============ Draw Dolphin ============
    glEnable(GL_TEXTURE_2D);
    glPushMatrix(); //5
    glTranslatef(-1.1f, -0.5f, -1.0f);
    glRotatef(90, 0.0f, 1.0f, 0.0f);
    glTranslatef(1.0f, dolJump, dolMove);
    glRotatef(dolFace, 0.0f, 1.0f, 0.0f);
    glRotatef(dolRot, 1.0f, 0.0f, 0.0f);
    glRotatef(20, 1.0f, 0.0f, 0.0f);drawObj(dolphin_vertices, dolphin_faces, num_dolphin_f, DOLPHIN_TEXTURE);
    glPopMatrix();  //5

    // ============ Draw Main character ============
    glDisable(GL_TEXTURE_2D);
    if (nShadow == 0)
    {
        glColor4f(mainColor.r, mainColor.g, mainColor.b, 1.0f);
        if (rasterizeMode == 0) {
            glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
            glPointSize(3.0);
        }
        else if (rasterizeMode == 1)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        else if (rasterizeMode == 2)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
    glPushMatrix(); //2
    if (nShadow == 0)
    {
        // Main character alone will be specular
        glMaterialfv(GL_FRONT, GL_SPECULAR, fBrightLight);
    }
    glTranslatef(0.0f, 0.01f, 0.0f);
    glRotatef(yRot, 0.0f, 1.0f, 0.0f);
    if(sw_boundingbox){
        drawBoundingBox();  // Draw bounding box
    }
    drawObj(main_vertices, main_faces, num_main_f, -1);
    glPopMatrix();  //2

    glEnable(GL_TEXTURE_2D);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glPopMatrix();  //0
}

// Called to draw scene
void RenderScene(void)
{
    static GLfloat seamove = -22.0f;         // Translate z for sea/sand animation
    static GLfloat seamove_toadd = 0.005f;
    if (seamove > -21.5 || seamove < -22.5) {
        seamove_toadd *= -1;
    }
    seamove += seamove_toadd;
    
    // Clear the window with current clearing color
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    glPushMatrix();
    //frameCamera.ApplyCameraTransform();

    //ApplyCameraTransform(frameCamera);


    // Position light before any other transformations
    glLightfv(GL_LIGHT0, GL_POSITION, fLightPos);

    // Draw the ground
    glColor3f(1.0f, 1.0f, 1.0f);
    glPushMatrix();
    glTranslatef(0.0f, 0.01f, 15.0f+seamove);
    DrawGround(SEA_TEXTURE, 5.0f, 5.0f);
    glPopMatrix();
    
    glPushMatrix();
    glTranslatef(-0.5f, 0.0f, -2.0f);
    DrawGround(SAND_TEXTURE, 1.5f, 1.0f);
    glPopMatrix();

    // Draw shadows first
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glDisable(GL_TEXTURE_2D);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_STENCIL_TEST);
    glPushMatrix();
    glMultMatrixf(mShadowMatrix);
    DrawInhabitants(1);
    glPopMatrix();
    glDisable(GL_STENCIL_TEST);
    glDisable(GL_BLEND);
    glEnable(GL_LIGHTING);
    glEnable(GL_TEXTURE_2D);
    glEnable(GL_DEPTH_TEST);

    // Draw inhabitants normally
    DrawInhabitants(0);

    glPopMatrix();

    // Do the buffer Swap
    glutSwapBuffers();
}

void TimerFunction(int value)
{
    if (value == 0) return;

    timer_cnt++;
    timer_cnt %= 256;
    // Redraw the scene with new coordinates
    glutPostRedisplay();
    glutTimerFunc(10, TimerFunction, timer_flag);
}

void ChangeSize(int w, int h)
{
    GLfloat fAspect;

    // Prevent a divide by zero, when window is too short
    // (you cant make a window of zero width).
    if (h == 0)
        h = 1;

    glViewport(0, 0, w, h);

    fAspect = (GLfloat)w / (GLfloat)h;

    // Reset the coordinate system before modifying
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    // Set the clipping volume
    gluPerspective(35.0f, fAspect, 1.0f, 50.0f);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void myKeyboard(unsigned char key, int x, int y) {
    switch (key) {
    case 'p':
        if (timer_flag == 0) {
            timer_flag = 1;
            std::cout << "Play" << endl;
            glutTimerFunc(10, TimerFunction, timer_flag);
        }
        else {
            timer_flag = 0;
            std::cout << "Pause" << endl;
        }
        break;
    default:
        break;
    }

    glutPostRedisplay();
}

int main(int argc, char* argv[])
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_STENCIL);
    glutInitWindowSize(800, 600);
    glutCreateWindow("Final 110598078");

    buildPopupMenu();
    readFile(objFiles[4], dolphin_vertices, dolphin_faces, num_dolphin_v, num_dolphin_f, 1.0f);
    readFile(objFiles[5], star_vertices, star_faces, num_star_v, num_star_f, 0.02f);
    cout << "In main." << endl;
    glutReshapeFunc(ChangeSize);
    glutDisplayFunc(RenderScene);
    
    glutKeyboardFunc(myKeyboard);

    SetupRC();
    glutTimerFunc(33, TimerFunction, timer_flag);

    glutMainLoop();

    ShutdownRC();
    glDeleteTextures(NUM_TEXTURES, textureObjects);
    return 0;
}

void clearMain() {          // clear vector of main character
    main_vertices.clear();
    main_faces.clear();
    objmax.x = 0;
    objmax.y = 0;
    objmax.z = 0;
    objmin.x = 0;
    objmin.y = 0;
    objmin.z = 0;
    num_main_v = num_main_f = 0;
}

// ====================================== Popup ============================================
void fileSelect(int option) {
    bool fileStat = false;
    if (option == 4) {
        clearMain();
        string inputObjPath; // input file path from terminal
        bool b = false;
        do {
            std::cout << endl << "Input file path:";
            getline(cin, inputObjPath);
            if (inputObjPath.compare("cancel") == 0) {
                b = true;
                break;
            }
            //fileStat = readFile(inputObjPath, 0);
            fileStat = readFile(inputObjPath, main_vertices, main_faces, num_main_v, num_main_f, 0);
        } while (!fileStat);
        if (b) { 
            cout << "Canceled." << endl;
            return; // cancel input obj file
        }
        std::cout << "Input path: " << inputObjPath << endl;
    }
    //else {
    else if(option != curr_obj_id) {
        clearMain();
        curr_obj_id = option;
        std::cout << endl << "Input path: " << objFiles[option] << endl;
        //fileStat = readFile(objFiles[option], 0);
        fileStat = readFile(objFiles[option], main_vertices, main_faces, num_main_v, num_main_f, 0);
    }
    glutPostRedisplay();
}

void renderSelect(int option)
{
    rasterizeMode = option;
    glutPostRedisplay();
}

void colorSelect(int option)
{
    if (option == 0) {
        // white
        mainColor.r = 1.0f; 
        mainColor.g = 1.0f;
        mainColor.b = 1.0f;
    }
    else if (option == 1) {
        // red
        mainColor.r = 1.0f;
        mainColor.g = 0.0f;
        mainColor.b = 0.0f;
    }
    else if (option == 2) {
        // green
        mainColor.r = 0.2f;
        mainColor.g = 1.0f;
        mainColor.b = 0.0f;
    }
    else if (option == 3) {
        // blue
        mainColor.r = 0.2f;
        mainColor.g = 0.2f;
        mainColor.b = 1.0f;
    }
    else if (option == 4) {
        // black
        mainColor.r = 0.0f;
        mainColor.g = 0.0f;
        mainColor.b = 0.0f;
    }
    else if (option == 5) {
        // generate random color
        srand((unsigned)time(0));        // for random colors
        rand(); // ignore the first random number
        mainColor.r = (float)rand() / RAND_MAX;
        mainColor.g = (float)rand() / RAND_MAX;
        mainColor.b = (float)rand() / RAND_MAX;
    }
    glutPostRedisplay();
}

void menuSelect(int option)
{
    sw_boundingbox = !sw_boundingbox;
    glutPostRedisplay();
}

void buildPopupMenu()
{
    int choose_file = glutCreateMenu(fileSelect);
    glutAddMenuEntry("gourd", 0);
    glutAddMenuEntry("octahedron", 1);
    glutAddMenuEntry("teapot", 2);
    glutAddMenuEntry("teddy", 3);
    glutAddMenuEntry("Input from terminal", 4);

    int choose_render = glutCreateMenu(renderSelect);
    glutAddMenuEntry("Point", 0);
    glutAddMenuEntry("Line", 1);
    glutAddMenuEntry("Polygon", 2);

    int choose_color = glutCreateMenu(colorSelect);
    glutAddMenuEntry("White", 0);
    glutAddMenuEntry("Black", 4);
    glutAddMenuEntry("Red", 1);
    glutAddMenuEntry("Green", 2);
    glutAddMenuEntry("Blue", 3);
    glutAddMenuEntry("Random", 5);

    int main_menu = glutCreateMenu(menuSelect);
    glutAddSubMenu("Files", choose_file);
    glutAddSubMenu("Render Mode", choose_render);
    glutAddSubMenu("Color", choose_color);
    glutAddMenuEntry("Bounding Box On/Off", sw_boundingbox);
    glutAttachMenu(GLUT_RIGHT_BUTTON);
}

// ====================================== Read file ============================================
void vertex_preprocess(vector<vertex>& v, int n, float dx, float dy, float dz, float scale, bool is_main) {
    //float xmax = 0, xmin = 0, ymax = 0, ymin = 0, zmax = 0, zmin = 0;
    for (int i = 0; i < n; i++) {
        v[i].x += dx;
        v[i].y += dy;
        v[i].z += dz;
        v[i].x *= scale;
        v[i].y *= scale;
        v[i].z *= scale;
        /*if (v[i].x > xmax) xmax = v[i].x;
        if (v[i].x < xmin) xmin = v[i].x;
        if (v[i].y > ymax) ymax = v[i].y;
        if (v[i].y < ymin) ymin = v[i].y;
        if (v[i].z > zmax) zmax = v[i].z;
        if (v[i].z < zmin) zmin = v[i].z;*/
    }
}

bool readFile(string objpath, vector<vertex> &v, vector<face> &f, int &nv, int &nf, float objscale) {
    /*FILE* file = fopen(objpath.c_str(), "r");
    if (file == NULL) {
        printf("File not found!\n");
        return false;
    }*/
    ifstream file(objpath);
    if (file.fail()) {
        printf("File not found!\n");
        return false;
    }
    string line;
    string deli = " ";
    float xmax = 0, xmin = 0, ymax = 0, ymin = 0, zmax = 0, zmin = 0;
    while (getline(file, line)) {
        string word;
        int start = 0;
        int end = line.find(deli, start);
        if (end != -1 && line.substr(start, end - start) == "v") {
            // add vertex
            vertex temp;
            for (int i = 0; i < 3; i++) {
                start = end + deli.size();
                end = line.find(deli, start);
                if (i == 0) {
                    temp.x = stof(line.substr(start, end - start));
                    if (temp.x > xmax) xmax = temp.x;
                    if (temp.x < xmin) xmin = temp.x;
                }
                else if (i == 1) {
                    temp.y = stof(line.substr(start, end - start));
                    if (temp.y > ymax) ymax = temp.y;
                    if (temp.y < ymin) ymin = temp.y;
                }
                else if (i == 2) {
                    temp.z = stof(line.substr(start, end - start));
                    if (temp.z > zmax) zmax = temp.z;
                    if (temp.z < zmin) zmin = temp.z;
                }
            }
            v.push_back(temp);
            nv++;
        }
        else if (end != -1 && line.substr(start, end - start) == "f") {
            // add face
            face temp;
            for (int i = 0; i < 3; i++) {
                start = end + deli.size();
                end = line.find(deli, start);
                if (i == 0)
                    temp.v0 = stoi(line.substr(start, end - start)) - 1;
                else if (i == 1)
                    temp.v1 = stoi(line.substr(start, end - start)) - 1;
                else if (i == 2)
                    temp.v2 = stoi(line.substr(start, end - start)) - 1;
            }
            f.push_back(temp);
            nf++;
        }
        
    }
    cout << "File load complete." << endl;
    file.close();

    float range = max( xmax - xmin, ymax - ymin );
    range = max(range, zmax - zmin);
    bool is_main = false;
    if (objscale == 0) {    // main obj
        objscale = 1.0f / range;
        is_main = true;
    }
    vertex_preprocess(v, nv, -((xmax - xmin) / 2 + xmin), -((ymax - ymin) / 2 + ymin), -((zmax - zmin) / 2 + zmin), objscale, is_main);
    if (is_main) {

        objmax.x = (xmax - ((xmax - xmin) / 2 + xmin)) * objscale;
        objmin.x = (xmin - ((xmax - xmin) / 2 + xmin)) * objscale;
        objmax.y = (ymax - ((ymax - ymin) / 2 + ymin)) * objscale;
        objmin.y = (ymin - ((ymax - ymin) / 2 + ymin)) * objscale;
        objmax.z = (zmax - ((zmax - zmin) / 2 + zmin)) * objscale;
        objmin.z = (zmin - ((zmax - zmin) / 2 + zmin)) * objscale;
    }
    return true;
}

