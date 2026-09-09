
#include "Render3d.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>

// ============= Shader =============
static const char* kVS_Volume = R"GLSL(
#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 uView; uniform mat4 uProj;
out vec3 vObjPos;
void main(){ vObjPos = aPos; gl_Position = uProj * uView * vec4(aPos,1.0); }
)GLSL";

static const char* kFS_Volume = R"GLSL(
#version 330 core
in vec3 vObjPos; out vec4 FragColor;
uniform sampler3D uVolume; uniform vec3 uCamPosObj;
uniform float uStep, uDensityScale, uBrightness;
bool intersectAABB(vec3 ro, vec3 rd, out float t0, out float t1){
    vec3 inv = 1.0/rd;
    vec3 tbot = (-1.0-ro)*inv, ttop=(1.0-ro)*inv;
    vec3 tmin=min(tbot,ttop), tmax=max(tbot,ttop);
    t0=max(max(tmin.x,tmin.y),tmin.z); t1=min(min(tmax.x,tmax.y),tmax.z);
    return t1>max(t0,0.0);
}
void main(){
    vec3 ro=uCamPosObj; vec3 rd=normalize(vObjPos-ro);
    float t0,t1; if(!intersectAABB(ro,rd,t0,t1)) discard;
    float t=max(t0,0.0),T=1.0;
    for(int i=0;i<1024 && t<t1 && T>0.01;i++){
        vec3 p=ro+rd*t; vec3 uvw=p*0.5+0.5;
        float d=texture(uVolume,uvw).r;
        float a=1.0-exp(-d*uDensityScale*uStep);
        a=clamp(a,0.0,1.0); T*=(1.0-a); t+=uStep;
    }
    vec3 color=mix(vec3(1.0),vec3(0.35,0.35,0.35),1.0-T)*uBrightness;
    FragColor=vec4(color,1.0);
}
)GLSL";

static const char* kVS_Wire = R"GLSL(
#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 uView; uniform mat4 uProj;
void main(){ gl_Position=uProj*uView*vec4(aPos,1.0); }
)GLSL";

static const char* kFS_Wire = R"GLSL(
#version 330 core
out vec4 FragColor;
void main(){ FragColor=vec4(0.2,0.6,1.0,1.0); }
)GLSL";

// ============= Utility =============
static unsigned int compile(GLenum t,const char*src){
    unsigned int s=glCreateShader(t); glShaderSource(s,1,&src,nullptr); glCompileShader(s);
    int ok; glGetShaderiv(s,GL_COMPILE_STATUS,&ok);
    if(!ok){ int len; glGetShaderiv(s,GL_INFO_LOG_LENGTH,&len);
        std::string log(len,'\0'); glGetShaderInfoLog(s,len,nullptr,log.data());
        throw std::runtime_error(log); }
    return s;
}
unsigned int VolumeRenderer::makeShaderProgram_(const char*vs,const char*fs){
    unsigned int V=compile(GL_VERTEX_SHADER,vs),F=compile(GL_FRAGMENT_SHADER,fs);
    unsigned int P=glCreateProgram(); glAttachShader(P,V); glAttachShader(P,F); glLinkProgram(P);
    glDeleteShader(V); glDeleteShader(F); return P;
}

// ============= Input Callbacks =============
static void sCursorCB(GLFWwindow*w,double x,double y){
    auto*self=reinterpret_cast<VolumeRenderer*>(glfwGetWindowUserPointer(w)); if(self) self->onCursor_(static_cast<Real>(x),static_cast<Real>(y));
}
static void sMouseCB(GLFWwindow*w,int b,int a,int m){
    auto*self=reinterpret_cast<VolumeRenderer*>(glfwGetWindowUserPointer(w)); if(self) self->onMouse_(b,a,m);
}
static void sScrollCB(GLFWwindow*w,double x,double y){
    auto*self=reinterpret_cast<VolumeRenderer*>(glfwGetWindowUserPointer(w)); if(self) self->onScroll_(static_cast<Real>(x),static_cast<Real>(y));
}

// ============= Core Implementation =============
VolumeRenderer::VolumeRenderer(){}
VolumeRenderer::~VolumeRenderer(){ if(window_) glfwTerminate(); }

bool VolumeRenderer::initWindow(int width,int height,const std::string&title){
    if(!glfwInit()) return false;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow*win=glfwCreateWindow(width,height,title.c_str(),nullptr,nullptr);
    if(!win) return false;
    glfwMakeContextCurrent(win);
    if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) return false;
    window_=win; win_w_=width; win_h_=height;
    glfwSetWindowUserPointer(win,this);
    setupInputCallbacks_();
    initGL_(); return true;
}

void VolumeRenderer::setupInputCallbacks_(){
    GLFWwindow*win=reinterpret_cast<GLFWwindow*>(window_);
    glfwSetCursorPosCallback(win,sCursorCB);
    glfwSetMouseButtonCallback(win,sMouseCB);
    glfwSetScrollCallback(win,sScrollCB);
}

void VolumeRenderer::onCursor_(Real x,Real y){
    if(!dragging_){ last_x_=x; last_y_=y; return;}
    Real dx=x-last_x_, dy=y-last_y_; last_x_=x; last_y_=y;
    cam_yaw_+=float(dx)*sensitivity_;
    cam_pitch_-=float(dy)*sensitivity_;
    cam_pitch_=glm::clamp(cam_pitch_,-85.0f,85.0f);
}
void VolumeRenderer::onMouse_(int button,int action,int){ if(button==GLFW_MOUSE_BUTTON_LEFT) dragging_=(action==GLFW_PRESS);}
void VolumeRenderer::onScroll_(Real,Real yoff){ cam_radius_*=(yoff>0?0.9f:1.1f); cam_radius_=glm::clamp(cam_radius_,1.5f,20.0f); }

// ============= Geometry & GL Init =============
void VolumeRenderer::initGL_(){
    buildCube_(); buildWireCube_();
    prog_volume_=makeShaderProgram_(kVS_Volume,kFS_Volume);
    prog_wire_=makeShaderProgram_(kVS_Wire,kFS_Wire);
    glGenTextures(1,&tex3d_);
    glBindTexture(GL_TEXTURE_3D,tex3d_);
    glTexParameteri(GL_TEXTURE_3D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D,GL_TEXTURE_WRAP_R,GL_CLAMP_TO_EDGE);
}

void VolumeRenderer::buildCube_(){
    const float v[]={-1,-1,-1,1,-1,-1,1,1,-1,-1,1,-1,-1,-1,1,1,-1,1,1,1,1,-1,1,1};
    const unsigned int idx[]={0,1,2,2,3,0,4,5,6,6,7,4,0,4,7,7,3,0,1,5,6,6,2,1,3,2,6,6,7,3,0,1,5,5,4,0};
    glGenVertexArrays(1,&vao_); glGenBuffers(1,&vbo_); glGenBuffers(1,&ebo_);
    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER,vbo_); glBufferData(GL_ARRAY_BUFFER,sizeof(v),v,GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,ebo_); glBufferData(GL_ELEMENT_ARRAY_BUFFER,sizeof(idx),idx,GL_STATIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,3*sizeof(float),(void*)0);
}

void VolumeRenderer::buildWireCube_(){
    const float e[]={-1,-1,-1,1,-1,-1, 1,-1,-1,1,1,-1, 1,1,-1,-1,1,-1, -1,1,-1,-1,-1,-1,
                     -1,-1,1,1,-1,1, 1,-1,1,1,1,1, 1,1,1,-1,1,1, -1,1,1,-1,-1,1,
                     -1,-1,-1,-1,-1,1, 1,-1,-1,1,-1,1, 1,1,-1,1,1,1, -1,1,-1,-1,1,1};
    glGenVertexArrays(1,&vao_lines_);
    glGenBuffers(1,&vbo_lines_);
    glBindVertexArray(vao_lines_);
    glBindBuffer(GL_ARRAY_BUFFER,vbo_lines_);
    glBufferData(GL_ARRAY_BUFFER,sizeof(e),e,GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,3*sizeof(float),(void*)0);
}

// ============= Per-frame Rendering =============
// void VolumeRenderer::draw_frame(const Array3D<Real>& density){
//     vol_res_=glm::ivec3((int)density.nx(),(int)density.ny(),(int)density.nz());
//     glBindTexture(GL_TEXTURE_3D,tex3d_);
//     glTexImage3D(GL_TEXTURE_3D,0,GL_R16F,vol_res_.x,vol_res_.y,vol_res_.z,0,GL_RED,GL_FLOAT,density.data());
//
//     float yaw=glm::radians(cam_yaw_),pitch=glm::radians(cam_pitch_);
//     glm::vec3 cam(cam_radius_*cosf(yaw)*cosf(pitch),cam_radius_*sinf(pitch),cam_radius_*sinf(yaw)*cosf(pitch));
//     glm::mat4 view=glm::lookAt(cam,glm::vec3(0),glm::vec3(0,1,0));
//     glm::mat4 proj=glm::perspective(glm::radians(45.0f),(float)win_w_/win_h_,0.01f,50.0f);
//
//     glViewport(0,0,win_w_,win_h_);
//     glClearColor(0.95f,0.95f,0.98f,1.0f);
//     glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
//
//     glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
//     glUseProgram(prog_volume_);
//     glUniformMatrix4fv(glGetUniformLocation(prog_volume_,"uView"),1,GL_FALSE,glm::value_ptr(view));
//     glUniformMatrix4fv(glGetUniformLocation(prog_volume_,"uProj"),1,GL_FALSE,glm::value_ptr(proj));
//     glUniform3fv(glGetUniformLocation(prog_volume_,"uCamPosObj"),1,glm::value_ptr(cam));
//     glUniform1f(glGetUniformLocation(prog_volume_,"uStep"),params_.step);
//     glUniform1f(glGetUniformLocation(prog_volume_,"uDensityScale"),params_.density_scale);
//     glUniform1f(glGetUniformLocation(prog_volume_,"uBrightness"),params_.brightness);
//     glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_3D,tex3d_);
//     glUniform1i(glGetUniformLocation(prog_volume_,"uVolume"),0);
//     glBindVertexArray(vao_); glDrawElements(GL_TRIANGLES,36,GL_UNSIGNED_INT,0);
//
//     if(params_.draw_wire){
//         glDisable(GL_BLEND); glEnable(GL_DEPTH_TEST);
//         glUseProgram(prog_wire_);
//         glUniformMatrix4fv(glGetUniformLocation(prog_wire_,"uView"),1,GL_FALSE,glm::value_ptr(view));
//         glUniformMatrix4fv(glGetUniformLocation(prog_wire_,"uProj"),1,GL_FALSE,glm::value_ptr(proj));
//         glBindVertexArray(vao_lines_);
//         glLineWidth(1.5f);
//         glDrawArrays(GL_LINES,0,24);
//         glLineWidth(1.0f);
//     }
//
// }

void VolumeRenderer::draw_frame(const Array3D<Real>& density) {
    // 1️⃣ 获取体素分辨率
    vol_res_ = glm::ivec3((int)density.nx(), (int)density.ny(), (int)density.nz());

    // 2️⃣ 将 Real → float（OpenGL 不支持直接上传 Real 数据）
    std::vector<float> density_f(vol_res_.x * vol_res_.y * vol_res_.z);
    for (size_t i = 0; i < density_f.size(); ++i)
        density_f[i] = static_cast<float>(density.data()[i]);

    // 3️⃣ 上传 3D 纹理数据（推荐使用 GL_R32F 以保持足够精度）
    glBindTexture(GL_TEXTURE_3D, tex3d_);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F,
                 vol_res_.x, vol_res_.y, vol_res_.z,
                 0, GL_RED, GL_FLOAT, density_f.data());

    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    // 4️⃣ 构造摄像机视图与投影矩阵
    float yaw = glm::radians(cam_yaw_);
    float pitch = glm::radians(cam_pitch_);
    glm::vec3 cam(
        cam_radius_ * cosf(yaw) * cosf(pitch),
        cam_radius_ * sinf(pitch),
        cam_radius_ * sinf(yaw) * cosf(pitch)
    );

    glm::mat4 view = glm::lookAt(cam, glm::vec3(0.0f), glm::vec3(0, 1, 0));
    glm::mat4 proj = glm::perspective(
        glm::radians(45.0f),
        (float)win_w_ / (float)win_h_,
        0.01f,
        50.0f
    );

    // 5️⃣ 清除并设置视口
    glViewport(0, 0, win_w_, win_h_);
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.95f, 0.95f, 0.98f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // 6️⃣ 渲染体积
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glUseProgram(prog_volume_);

    glUniformMatrix4fv(glGetUniformLocation(prog_volume_, "uView"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(prog_volume_, "uProj"), 1, GL_FALSE, glm::value_ptr(proj));
    glUniform3fv(glGetUniformLocation(prog_volume_, "uCamPosObj"), 1, glm::value_ptr(cam));
    glUniform1f(glGetUniformLocation(prog_volume_, "uStep"), params_.step);
    glUniform1f(glGetUniformLocation(prog_volume_, "uDensityScale"), params_.density_scale);
    glUniform1f(glGetUniformLocation(prog_volume_, "uBrightness"), params_.brightness);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, tex3d_);
    glUniform1i(glGetUniformLocation(prog_volume_, "uVolume"), 0);

    glBindVertexArray(vao_);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);

    // 7️⃣ 可选线框绘制
    if (params_.draw_wire) {
        glDisable(GL_BLEND);
        glEnable(GL_DEPTH_TEST);
        glUseProgram(prog_wire_);
        glUniformMatrix4fv(glGetUniformLocation(prog_wire_, "uView"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(prog_wire_, "uProj"), 1, GL_FALSE, glm::value_ptr(proj));
        glBindVertexArray(vao_lines_);
        glLineWidth(1.5f);
        glDrawArrays(GL_LINES, 0, 24);
        glLineWidth(1.0f);
    }

    glBindVertexArray(0);
    glUseProgram(0);
}




bool VolumeRenderer::should_close(){
    return glfwWindowShouldClose(reinterpret_cast<GLFWwindow*>(window_));
}
void VolumeRenderer::poll_events(){
    glfwSwapBuffers(reinterpret_cast<GLFWwindow*>(window_));
    glfwPollEvents();
}

void VolumeRenderer::renderParticles(const std::vector<RenderParticle>& particles) {
    if (particles.empty()) return;

    static unsigned int vao = 0, vbo = 0, prog = 0;

    // ============ 初始化一次 =============
    if (vao == 0) {
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);

        const char* vs = R"GLSL(
            #version 330 core
            layout(location=0) in vec3 aPos;
            layout(location=1) in float aSpeed;
            out float vSpeed;
            uniform mat4 uView;
            uniform mat4 uProj;
            void main(){
                vSpeed = aSpeed;
                gl_Position = uProj * uView * vec4(aPos, 1.0);
                gl_PointSize = clamp(aSpeed * 4.0 + 2.0, 2.0, 8.0);
            }
        )GLSL";

        const char* fs = R"GLSL(
            #version 330 core
            in float vSpeed;
            out vec4 FragColor;
            void main(){
                float s = clamp(vSpeed * 0.3, 0.0, 1.0);
                vec3 color;
                if (s < 0.5)
                    color = mix(vec3(0.0,0.3,1.0), vec3(0.0,1.0,0.0), s*2.0);
                else
                    color = mix(vec3(0.0,1.0,0.0), vec3(1.0,0.0,0.0), (s-0.5)*2.0);
                FragColor = vec4(color, 1.0);
            }
        )GLSL";

        unsigned int vsid = compile(GL_VERTEX_SHADER, vs);
        unsigned int fsid = compile(GL_FRAGMENT_SHADER, fs);
        prog = glCreateProgram();
        glAttachShader(prog, vsid);
        glAttachShader(prog, fsid);
        glLinkProgram(prog);
        glDeleteShader(vsid);
        glDeleteShader(fsid);
    }

    // ============ 上传数据 ============
    struct PData { glm::vec3 pos; float speed; };
    std::vector<PData> buf(particles.size());
    for (size_t i = 0; i < particles.size(); ++i) {
        // 把 [0,1] 坐标映射到 [-1,1]
        buf[i].pos = (particles[i].pos - glm::vec3(0.5f)) * 2.0f;
        buf[i].speed = particles[i].speed;
    }

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, buf.size() * sizeof(PData), buf.data(), GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(PData), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(PData), (void*)offsetof(PData, speed));

    // ============ 相机矩阵（与体渲染一致） ============
    float yaw = glm::radians(cam_yaw_), pitch = glm::radians(cam_pitch_);
    glm::vec3 cam(cam_radius_*cosf(yaw)*cosf(pitch),
                  cam_radius_*sinf(pitch),
                  cam_radius_*sinf(yaw)*cosf(pitch));
    glm::mat4 view = glm::lookAt(cam, glm::vec3(0,0,0), glm::vec3(0,1,0));
    glm::mat4 proj = glm::perspective(glm::radians(45.0f),
                                      (float)win_w_ / win_h_, 0.01f, 50.0f);

    // ============ 绘制 ============
    glUseProgram(prog);
    glUniformMatrix4fv(glGetUniformLocation(prog, "uView"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(prog, "uProj"), 1, GL_FALSE, glm::value_ptr(proj));

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glDisable(GL_DEPTH_TEST);

    glDrawArrays(GL_POINTS, 0, (GLsizei)particles.size());

    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
}

// void VolumeRenderer::drawPartices(const std::vector<Smoke_Particles_3d>& smoke_particles, float particle_size, const glm::vec3 particle_color) {
//     if (smoke_particles.empty()) return;
//     // glClearColor(0.95f,0.95f,0.98f,1.0f);
//     // glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
//     // 着色器代码
//     const char* vs = R"GLSL(
//         #version 330 core
//         layout(location=0) in vec3 aPos;
//         out vec3 vPos;
//         uniform mat4 uView;
//         uniform mat4 uProj;
//         void main(){
//             vPos = aPos;
//             gl_Position = uProj * uView * vec4(aPos, 1.0);
//             gl_PointSize = 10.0;  // 固定粒子大小，可以根据需要调整
//         }
//     )GLSL";
//
//     const char* fs = R"GLSL(
//         #version 330 core
//         in vec3 vPos;
//         out vec4 FragColor;
//         uniform vec3 uParticleColor;
//         void main(){
//             FragColor = vec4(uParticleColor, 1.0);
//         }
//     )GLSL";
//
//     // 编译着色器并链接程序
//     unsigned int vsid = compile(GL_VERTEX_SHADER, vs);
//     unsigned int fsid = compile(GL_FRAGMENT_SHADER, fs);
//     unsigned int prog = glCreateProgram();
//     glAttachShader(prog, vsid);
//     glAttachShader(prog, fsid);
//     glLinkProgram(prog);
//     glDeleteShader(vsid);
//     glDeleteShader(fsid);
//
//     // 创建 VAO 和 VBO
//     unsigned int vao, vbo;
//     glGenVertexArrays(1, &vao);
//     glGenBuffers(1, &vbo);
//
//     struct PData { glm::vec3 pos; };
//     std::vector<PData> buf(smoke_particles.size());
//
//     // 设置粒子数据
//     for (size_t i = 0; i < smoke_particles.size(); ++i) {
//         buf[i].pos.x = smoke_particles[i].pos.vx;
//         buf[i].pos.y = smoke_particles[i].pos.vy;
//         buf[i].pos.z = smoke_particles[i].pos.vz;
//     }
//
//     // 绑定 VAO 和 VBO，上传数据
//     glBindVertexArray(vao);
//     glBindBuffer(GL_ARRAY_BUFFER, vbo);
//     glBufferData(GL_ARRAY_BUFFER, buf.size() * sizeof(PData), buf.data(), GL_DYNAMIC_DRAW);
//     glEnableVertexAttribArray(0);
//     glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(PData), (void*)0);
//
//     // 使用类内部的相机和投影矩阵
//     float yaw = glm::radians(cam_yaw_), pitch = glm::radians(cam_pitch_);
//     glm::vec3 cam(cam_radius_ * cosf(yaw) * cosf(pitch), cam_radius_ * sinf(pitch), cam_radius_ * sinf(yaw) * cosf(pitch));
//     glm::mat4 view = glm::lookAt(cam, glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
//     glm::mat4 proj = glm::perspective(glm::radians(45.0f), (float)win_w_ / win_h_, 0.01f, 50.0f);
//
//     // 使用着色器
//     glUseProgram(prog);
//     glUniformMatrix4fv(glGetUniformLocation(prog, "uView"), 1, GL_FALSE, glm::value_ptr(view));
//     glUniformMatrix4fv(glGetUniformLocation(prog, "uProj"), 1, GL_FALSE, glm::value_ptr(proj));
//
//     // 传递粒子颜色到着色器
//     glUniform3fv(glGetUniformLocation(prog, "uParticleColor"), 1, glm::value_ptr(particle_color));
//
//     // 启用混合，并禁用深度测试（粒子通常会在前面渲染）
//     glEnable(GL_BLEND);
//     glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
//     glEnable(GL_PROGRAM_POINT_SIZE);
//     glDisable(GL_DEPTH_TEST);
//
//     // 绘制粒子
//     glDrawArrays(GL_POINTS, 0, (GLsizei)smoke_particles.size());
//
//     // 恢复默认状态
//     glDisable(GL_BLEND);
//     glEnable(GL_DEPTH_TEST);
// }
void VolumeRenderer::drawParticles(const std::vector<Smoke_Particles_3d>& smoke_particles, float particle_size, const glm::vec3 particle_color) {
    if (smoke_particles.empty()) return;

    // 着色器代码
    const char* vs = R"GLSL(
        #version 330 core
        layout(location=0) in vec3 aPos;
        out vec3 vPos;
        uniform mat4 uView;
        uniform mat4 uProj;
        void main(){
            vPos = aPos;
            gl_Position = uProj * uView * vec4(aPos, 1.0);
            gl_PointSize = 0.01;  // 固定粒子大小，可以根据需要调整
        }
    )GLSL";

    const char* fs = R"GLSL(
        #version 330 core
        in vec3 vPos;
        out vec4 FragColor;
        uniform vec3 uParticleColor;
        void main(){
            FragColor = vec4(uParticleColor, 1.0);
        }
    )GLSL";

    // 编译着色器并链接程序
    unsigned int vsid = compile(GL_VERTEX_SHADER, vs);
    unsigned int fsid = compile(GL_FRAGMENT_SHADER, fs);
    unsigned int prog = glCreateProgram();
    glAttachShader(prog, vsid);
    glAttachShader(prog, fsid);
    glLinkProgram(prog);
    glDeleteShader(vsid);
    glDeleteShader(fsid);

    // 创建 VAO 和 VBO
    unsigned int vao, vbo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    struct PData { glm::vec3 pos; };
    int _size =0;
    for (auto&p:smoke_particles)if (!p.deleted)++_size;
    std::vector<PData> buf(_size);

    // 设置粒子数据
    size_t buf_i=0;
    for (size_t i = 0; i < smoke_particles.size(); ++i) {
        if (smoke_particles[i].deleted)continue;
        // 将粒子位置从[0, 1]坐标系映射到[-1, 1]范围
        buf[buf_i].pos.x = smoke_particles[i].pos.vx;
        buf[buf_i].pos.y = smoke_particles[i].pos.vy;
        buf[buf_i].pos.z = smoke_particles[i].pos.vz;
        glm::vec3 normalized_pos = ( buf[buf_i].pos - glm::vec3(0.5f)) * 2.0f; // 归一化
        buf[buf_i++].pos = normalized_pos;  // 设置粒子位置
    }

    // 绑定 VAO 和 VBO，上传数据
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, buf.size() * sizeof(PData), buf.data(), GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(PData), (void*)0);

    // 使用类内部的相机和投影矩阵
    float yaw = glm::radians(cam_yaw_), pitch = glm::radians(cam_pitch_);
    glm::vec3 cam(cam_radius_ * cosf(yaw) * cosf(pitch), cam_radius_ * sinf(pitch), cam_radius_ * sinf(yaw) * cosf(pitch));
    glm::mat4 view = glm::lookAt(cam, glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
    glm::mat4 proj = glm::perspective(glm::radians(45.0f), (float)win_w_ / win_h_, 0.01f, 50.0f);

    // 使用着色器
    glUseProgram(prog);
    glUniformMatrix4fv(glGetUniformLocation(prog, "uView"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(prog, "uProj"), 1, GL_FALSE, glm::value_ptr(proj));

    // 传递粒子颜色到着色器
    glUniform3fv(glGetUniformLocation(prog, "uParticleColor"), 1, glm::value_ptr(particle_color));

    // 启用混合，并禁用深度测试（粒子通常会在前面渲染）
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glDisable(GL_DEPTH_TEST);

    // 绘制粒子
    glDrawArrays(GL_POINTS, 0, (GLsizei)_size);

    // 恢复默认状态
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
}
