#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/hash.hpp>
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include <iostream>
#include <unordered_map>

const uint32_t SCR_WIDTH = 800;
const uint32_t SCR_HEIGHT = 600;
const uint32_t RSM_EVAL_RES = 4;

#pragma region Data Structure
struct MeshObj
{
    uint32_t vao;
    uint32_t indexCount;
};
struct Vertex
{
    glm::vec3 position;
    glm::vec3 normal;
    bool operator==(const Vertex& other) const
    {
        return other.position == position && other.normal == normal;
    }
};
struct VertexHash
{
    size_t operator()(const Vertex& vertex) const
    {
        return ((std::hash<glm::vec3>()(vertex.position) ^ (std::hash<glm::vec3>()(vertex.normal) << 1)) >> 1);
    }
};
#pragma endregion

#pragma region Helper Functions
void loadObj(const char* path, std::vector<Vertex>& vertices, std::vector<uint32_t>& indices)
{
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string err;
    bool result;
#if WIN32
    result = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, path);
#endif
#if __APPLE__
    std::string warn;
    result = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, path);
#endif
    if (!result)
    {
        std::cout << "Failed to load model at " << path << std::endl;
    }
    std::unordered_map<Vertex, uint32_t, VertexHash> uniqueVertices;
    for (const auto& shape : shapes)
    {
        for (const auto& index : shape.mesh.indices)
        {
            Vertex v {};
            v.position = {
                attrib.vertices[3*index.vertex_index + 0],
                attrib.vertices[3*index.vertex_index + 1],
                attrib.vertices[3*index.vertex_index + 2]
            };
            v.normal = {
                attrib.normals[3*index.normal_index + 0],
                attrib.normals[3*index.normal_index + 1],
                attrib.normals[3*index.normal_index + 2]
            };
            if (uniqueVertices.count(v) == 0)
            {
                uniqueVertices[v] = (uint32_t) vertices.size();
                vertices.push_back(v);
            }
            indices.push_back(uniqueVertices[v]);
        }
    }
}
uint32_t createShaderProgram(const char* vertex_shader_source, const char* fragment_shader_source)
{
    const auto vertex_shader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex_shader, 1, &vertex_shader_source, nullptr);
    glCompileShader(vertex_shader);
    int success;
    char info_log[512];
    glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(vertex_shader, 512, nullptr, info_log);
        std::cout << "Failed to compile shader! " << info_log << std::endl;
    }

    const auto fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment_shader, 1, &fragment_shader_source, nullptr);
    glCompileShader(fragment_shader);
    glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(fragment_shader, 512, nullptr, info_log);
        std::cout << "Failed to compile shader! " << info_log << std::endl;
    }

    const auto program = glCreateProgram();
    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    glLinkProgram(program);
    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);
    return program;
}
MeshObj createVertexDataFromObj(const char* path)
{
    std::vector<Vertex> vertices {};
    std::vector<uint32_t> indices {};
    loadObj(path, vertices, indices);
    uint32_t vao, vbo, ebo;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex)*vertices.size(), vertices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), nullptr);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*) (sizeof(glm::vec3)));
    glEnableVertexAttribArray(1);
    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(uint32_t)*indices.size(), indices.data(), GL_STATIC_DRAW);
    MeshObj out{};
    out.vao = vao;
    out.indexCount = indices.size();
    return out;
}
void setupRsmFramebuffer(uint32_t& fbo, uint32_t* tex2d)
{
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glGenTextures(4, tex2d);
    float col_border[] = {0.0f, 0.0f, 0.0f, 1.0f};
    for (auto i = 0; i < 3; ++i)
    {
        glBindTexture(GL_TEXTURE_2D, tex2d[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, col_border);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, tex2d[i], 0);
    }
    uint32_t attachments[3] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2};
    glDrawBuffers(3, attachments);

    glBindTexture(GL_TEXTURE_2D, tex2d[3]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, SCR_WIDTH, SCR_HEIGHT, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, col_border);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, tex2d[3], 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        std::cout << "Failed to complete framebuffer!" << std::endl;
    }
}
void setupRsmEvalFramebuffer(uint32_t& fbo, uint32_t& tex2d)
{
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glGenTextures(1, &tex2d);
    glBindTexture(GL_TEXTURE_2D, tex2d);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, SCR_WIDTH / RSM_EVAL_RES, SCR_HEIGHT / RSM_EVAL_RES, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex2d, 0);
    glDrawBuffer(GL_COLOR_ATTACHMENT0);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        std::cout << "Failed to complete framebuffer!" << std::endl;
    }
}
uint32_t createQuad()
{
    const float quad_vertices[] = {
        -1.0f, -1.0f, 0.0f, 0.0f,
         1.0f, -1.0f, 1.0f, 0.0f,
        -1.0f,  1.0f, 0.0f, 1.0f,
        -1.0f,  1.0f, 0.0f, 1.0f,
         1.0f, -1.0f, 1.0f, 0.0f,
         1.0f,  1.0f, 1.0f, 1.0f
    };
    uint32_t vao, vbo;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad_vertices), quad_vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float)*4, nullptr);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(float)*4, (void*) (sizeof(float)*2));
    glEnableVertexAttribArray(1);
    return vao;
}
#pragma endregion

#pragma region Shader Source
const char* vs_obj = R"(
#version 330 core
layout (location = 0) in vec3 pos;
layout (location = 1) in vec3 normal;
uniform mat4 mat_mvp;
out VS_OUT
{
    vec3 pos;
    vec3 normal;
} vs_out;
void main()
{
    gl_Position = mat_mvp*vec4(pos, 1.);
    vs_out.pos = pos;
    vs_out.normal = normal;
}
)";
const char* fs_obj = R"(
#version 330 core
out vec4 fragColor;
uniform sampler2D tex2d_depth;
uniform sampler2D tex2d_e;
uniform mat4 mat_light_vp;
uniform mat4 mat_mvp;
uniform int useRgb;
in VS_OUT
{
    vec3 pos;
    vec3 normal;
} fs_in;
#define PI 3.14159265358979
float getShadow(vec3 pos, vec3 normal)
{
    vec4 cpos = mat_light_vp*vec4(pos, 1.);
    cpos.xyz /= cpos.w;
    cpos.xyz = cpos.xyz*.5 + .5;
    if (cpos.z >= 1.)
        return 1.;
    vec3 dir_light = normalize(vec3(1., 2., 2.));
    float bias = max(.01*(1. - dot(normal, dir_light)), .0);
    vec2 offset = 1./textureSize(tex2d_depth, 0);
    float shadow = 0.;
    for (int i = -1; i <= 1; ++i)
    {
        for (int j = -1; j <= 1; ++j)
        {
            float depth = texture(tex2d_depth, cpos.xy + vec2(i, j)*offset).r;
            shadow += cpos.z - depth > bias ? 0. : 1.;
        }
    }
    return shadow/9.;
}
vec3 getIndirect(vec3 pos, vec3 color)
{
    vec4 cpos = mat_light_vp*vec4(pos, 1.);
    cpos.xyz /= cpos.w;
    cpos.xyz = cpos.xyz*.5 + .5;
    if (cpos.z >= 1.)
        return vec3(0.);
    vec2 offset = 1./textureSize(tex2d_e, 0);
    vec3 e = vec3(0.);
    for (int i = -1; i <= 1; ++i)
    {
        for (int j = -1; j <= 1; ++j)
        {
            e += texture(tex2d_e, cpos.xy + vec2(i, j)*offset).rgb;
        }
    }
    return e/PI/9.;
}
void main()
{
    vec3 normal = fs_in.normal;
    vec3 ldir = normalize(vec3(1., 2., 2.));
    float nl = max(0., dot(normal, ldir));
    vec3 col = vec3(0.);
    float shadow = getShadow(fs_in.pos, fs_in.normal);
    vec3 c = useRgb == 1 ? fs_in.normal : vec3(.8);
    vec3 indirect = getIndirect(fs_in.pos, c);
    col = shadow*(c*nl + indirect*nl);
    fragColor = vec4(col, 1.);
}
)";
const char* vs_rsm = R"(
#version 330 core
layout (location = 0) in vec3 pos;
layout (location = 1) in vec3 normal;
uniform mat4 mat_light_mvp;
out VS_OUT
{
    vec3 pos;
    vec3 normal;
} vs_out;
void main()
{
    gl_Position = mat_light_mvp*vec4(pos, 1.);
    vs_out.pos = pos;
    vs_out.normal = normal;
}
)";
const char* fs_rsm = R"(
#version 330 core
layout (location = 0) out vec3 PositionBuffer;
layout (location = 1) out vec3 NormalBuffer;
layout (location = 2) out vec3 FluxBuffer;
uniform int useRgb;
in VS_OUT
{
    vec3 pos;
    vec3 normal;
} fs_in;
void main()
{
    PositionBuffer = fs_in.pos;
    NormalBuffer = fs_in.normal*.5 + .5;
    FluxBuffer = useRgb == 1 ? fs_in.normal : vec3(.8);
}
)";
const char* fs_rsm_eval = R"(
#version 330 core
out vec4 fragColor;
uniform sampler2D tex2d_position;
uniform sampler2D tex2d_normal;
uniform sampler2D tex2d_flux;
uniform sampler2D tex2d_depth;
uniform float time;
in VS_OUT
{
    vec2 uv;
} fs_in;
float seed;
#define PI 3.14159265358979
#define MAX_NUM 400
#define MAX_R .15
float rand01()
{
    return fract(sin(++seed/113.*dot(gl_FragCoord.xy, vec2(12.9898, 78.233)))*43758.5453123);
}
vec3 getEp()
{
    float r0 = rand01();
    float r1 = rand01();
    vec2 offset = MAX_R*r0*vec2(sin(2.*PI*r1), cos(2.*PI*r1));
    vec2 uv = fs_in.uv + offset;
    //float depth = texture(tex2d_depth, fs_in.uv).r;
    vec3 n = texture(tex2d_normal, fs_in.uv).rgb*2. - 1.;
    vec3 np = texture(tex2d_normal, uv).rgb*2. - 1;
    vec3 x = texture(tex2d_position, fs_in.uv).rgb;
    vec3 xp = texture(tex2d_position, uv).rgb;
    vec3 d = normalize(x - xp);
    vec3 ep = texture(tex2d_flux, uv).rgb*max(0., dot(np, d))*max(0., dot(n, -d))/max(1e-5, pow(length(x - xp), 4.));
    ep *= r0*r0;
    return ep;
}
void main()
{
    vec3 col = vec3(0.);
    seed = time;
    for (int i = 0; i < MAX_NUM; ++i)
    {
        col += getEp();
    }
    fragColor = vec4(col/float(MAX_NUM), 1.);
}
)";
const char* vs_quad = R"(
#version 330 core
layout (location = 0) in vec2 pos;
layout (location = 1) in vec2 uv;
out VS_OUT
{
    vec2 uv;
} vs_out;
void main()
{
    gl_Position = vec4(pos, 0., 1.);
    vs_out.uv = uv;
}
)";
const char* fs_quad = R"(
#version 330 core
out vec4 fragColor;
uniform sampler2D tex2d_in;
in VS_OUT
{
    vec2 uv;
} fs_in;
void main()
{
    vec3 col = vec3(0.);
    vec2 uv = fs_in.uv;
    col = texture(tex2d_in, uv).rgb;
    fragColor = vec4(col, 1.);
}
)";
#pragma endregion

int main()
{

#pragma region Initialization
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    auto* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "RSM", nullptr, nullptr);
    if (window == nullptr)
    {
        std::cout << "Failed to create window!" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    if (!gladLoadGL())
    {
        std::cout << "Failed to load gl!" << std::endl;
        glfwTerminate();
        return -1;
    }
#pragma endregion

    const auto mo_planes = createVertexDataFromObj("./planes.obj");
    const auto mo_eve = createVertexDataFromObj("./eve.obj");
    const auto program_obj = createShaderProgram(vs_obj, fs_obj);

    uint32_t fbo_rsm;
    uint32_t tex2d_rsm[4];
    setupRsmFramebuffer(fbo_rsm, tex2d_rsm);
    const auto program_rsm = createShaderProgram(vs_rsm, fs_rsm);

    uint32_t fbo_rsm_eval;
    uint32_t tex2d_rsm_eval;
    setupRsmEvalFramebuffer(fbo_rsm_eval, tex2d_rsm_eval);
    const auto program_rsm_eval = createShaderProgram(vs_quad, fs_rsm_eval);

    const auto vao_quad = createQuad();
    const auto program_quad = createShaderProgram(vs_quad, fs_quad); 

    glm::mat4 mat_m = glm::mat4(1.0f);
    glm::mat4 mat_v = glm::mat4(1.0f);
    glm::mat4 mat_p = glm::mat4(1.0f);
    glm::mat4 mat_mvp = glm::mat4(1.0f);
    mat_v = glm::lookAt(glm::vec3(5.0f, 3.0f, 5.0f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    mat_p = glm::perspective(glm::radians(30.0f), 4.0f/3.0f, 0.1f, 10.0f);
    mat_mvp = mat_p*mat_v;

    glm::mat4 mat_light_m = glm::mat4(1.0f);
    glm::mat4 mat_light_v = glm::mat4(1.0f);
    glm::mat4 mat_light_p = glm::mat4(1.0f);
    glm::mat4 mat_light_mvp = glm::mat4(1.0f);
    mat_light_v = glm::lookAt(glm::vec3(1.0f, 3.0f, 2.0f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    mat_light_p = glm::ortho(-3.0f, 3.0f, -2.25f, 2.25f, 0.1f, 10.0f);
    mat_light_mvp = mat_light_p*mat_light_v;
    glUseProgram(program_rsm);
    glUniformMatrix4fv(glGetUniformLocation(program_rsm, "mat_light_mvp"), 1, GL_FALSE, &mat_light_mvp[0][0]);
    glUseProgram(program_obj);
    glUniformMatrix4fv(glGetUniformLocation(program_obj, "mat_light_vp"), 1, GL_FALSE, &mat_light_mvp[0][0]);
    glUniformMatrix4fv(glGetUniformLocation(program_obj, "mat_mvp"), 1, GL_FALSE, &mat_mvp[0][0]);

    int framebuffer_width, framebuffer_height;
    glfwGetFramebufferSize(window, &framebuffer_width, &framebuffer_height);
//    auto last_time = glfwGetTime();
//    uint32_t num_frames = 0;
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glEnable(GL_DEPTH_TEST);
    while (!glfwWindowShouldClose(window))
    {
        /*
         * RSM generation pass
         */
        glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_rsm);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(program_rsm);
        // Draw eve
        glUniform1i(glGetUniformLocation(program_rsm, "useRgb"), 0);
        glBindVertexArray(mo_eve.vao);
        glDrawElements(GL_TRIANGLES, mo_eve.indexCount, GL_UNSIGNED_INT, nullptr);
        // Draw planes
        glUniform1i(glGetUniformLocation(program_rsm, "useRgb"), 1);
        glBindVertexArray(mo_planes.vao);
        glDrawElements(GL_TRIANGLES, mo_planes.indexCount, GL_UNSIGNED_INT, nullptr);

        /*
         * RSM evaluation pass
         */
        glViewport(0, 0, SCR_WIDTH/RSM_EVAL_RES, SCR_HEIGHT/RSM_EVAL_RES);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_rsm_eval);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(program_rsm_eval);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex2d_rsm[0]);
        glUniform1i(glGetUniformLocation(program_rsm_eval, "tex2d_position"), 0);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, tex2d_rsm[1]);
        glUniform1i(glGetUniformLocation(program_rsm_eval, "tex2d_normal"), 1);
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, tex2d_rsm[2]);
        glUniform1i(glGetUniformLocation(program_rsm_eval, "tex2d_flux"), 2);
        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_2D, tex2d_rsm[3]);
        glUniform1i(glGetUniformLocation(program_rsm_eval, "tex2d_depth"), 3);
        auto t = glfwGetTime();
        glUniform1f(glGetUniformLocation(program_rsm_eval, "t"), t);
        glBindVertexArray(vao_quad);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        /*
         * Actual render pass
         */
        glViewport(0, 0, framebuffer_width, framebuffer_height);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(program_obj);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex2d_rsm[3]);
        glUniform1i(glGetUniformLocation(program_obj, "tex2d_depth"), 0);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, tex2d_rsm_eval);
        glUniform1i(glGetUniformLocation(program_obj, "tex2d_e"), 1);
        // Draw eve
        glUniform1i(glGetUniformLocation(program_obj, "useRgb"), 0);
        glBindVertexArray(mo_eve.vao);
        glDrawElements(GL_TRIANGLES, mo_eve.indexCount, GL_UNSIGNED_INT, nullptr);
        // Draw planes
        glUniform1i(glGetUniformLocation(program_obj, "useRgb"), 1);
        glBindVertexArray(mo_planes.vao);
        glDrawElements(GL_TRIANGLES, mo_planes.indexCount, GL_UNSIGNED_INT, nullptr);

        /*
         * Debug pass
         */
        /*glViewport(0, 0, framebuffer_width, framebuffer_height);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(program_quad);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex2d_rsm_eval);
        glUniform1i(glGetUniformLocation(program_quad, "tex2d_in"), 0);
        glBindVertexArray(vao_quad);
        glDrawArrays(GL_TRIANGLES, 0, 6);*/

        /*
         * FPS counter
         */
        /*++num_frames;
        auto current_time = glfwGetTime();
        if (current_time - last_time > 1.0)
        {
            auto fps = 1000.0/double(num_frames);
            std::cout << (int) fps << std::endl;
            num_frames = 0;
            last_time += 1.0;
        }*/

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glfwTerminate();
    return 0;
}
