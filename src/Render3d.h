
#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <string>
#include "dfn.h"
struct RenderParticle {
    glm::vec3 pos;
    glm::vec3 vel;
    float     speed;
};
enum class MouseButton { Left=0, Right=1, Middle=2 };

class VolumeRenderer {
public:
    struct Params {
        float step = 0.01f;
        float density_scale = 8.0f;
        float brightness = 1.0f;
        bool  draw_wire = true;
    };

    VolumeRenderer();
    ~VolumeRenderer();

    bool initWindow(int width = 1280, int height = 720, const std::string& title = "Volume Renderer");

    // 新接口：直接每帧传入密度渲染
    void draw_frame(const Array3D<Real>& density);
    void poll_events();
    bool should_close();

    void setParams(const Params& p) { params_ = p; }

    // 鼠标控制
    void onCursor_(Real x, Real y);
    void onMouse_(int button, int action, int mods);
    void onScroll_(Real xoff, Real yoff);
    void drawParticles(const std::vector<Smoke_Particles_3d>& smoke_particles, float particle_size=0.01, const glm::vec3 particle_color = glm::vec3(0.9,0.9,0.9));
    void renderParticles(const std::vector<RenderParticle>& particles) ;
private:
    void initGL_();
    void buildCube_();
    void buildWireCube_();
    unsigned int makeShaderProgram_(const char* vs_src, const char* fs_src);

    void setupInputCallbacks_();

    int win_w_ = 1280, win_h_ = 720;
    void* window_ = nullptr;

    unsigned int vao_ = 0, vbo_ = 0, ebo_ = 0;
    unsigned int vao_lines_ = 0, vbo_lines_ = 0;
    unsigned int tex3d_ = 0;
    unsigned int prog_volume_ = 0, prog_wire_ = 0;

    glm::ivec3 vol_res_{0};
    Params params_{};

    float cam_radius_ = 3.5f;
    float cam_yaw_    = 45.0f;
    float cam_pitch_  = 25.0f;
    bool dragging_ = false;
    Real last_x_ = 0.0, last_y_ = 0.0;
    float sensitivity_ = 0.2f;
};
