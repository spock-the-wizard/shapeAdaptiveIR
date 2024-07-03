#pragma once

#include <unordered_map>
#include <psdr/psdr.h>
#include <psdr/edge/edge.h>
#include <psdr/emitter/emitter.h>

namespace psdr
{

typedef struct LayerIntersection {
    void reserve(int64_t size, int64_t depth=0);
    int64_t size;
    int64_t depth;
    Vector2fD uvs;
    IntC numIntersections;
    IntC shape_ids;
} LayerIntersection;

PSDR_CLASS_DECL_BEGIN(Scene, final, Object)
    friend class SceneLoader;

public:
    using ParamMap = std::unordered_map<std::string, const Object&>;

    Scene();
    ~Scene() override;

    void load_file(const char *file_name, bool auto_configure = true);
    void load_string(const char *scene_xml, bool auto_configure = true);

    
    template <bool ad>
    Vector3fD crossing_edge(Vector3f<ad> p,Vectorf<1,ad> r, Vector2f<ad> sample) const;
    template <bool ad>
    Intersection<ad> bounding_edge(Scene scene, Intersection<ad> its, Vectorf<1,ad> r, Vector8f<ad> sample) const;
    // Vector3fD bounding_edge(Intersection<ad> _p, Vectorf<1,ad> sphereRadius, Vector2f<ad> sample) const;

    void configure();
    bool is_ready() const;

    void setseed(const int seed) {
        m_seed = seed;
        // m_samplers[0].seed(m_seed);
    }

    void setlightposition(Vector3fD p){
        m_emitters[0]->setposition(p);
    }

    void setEdgeDistr(int edge_num) {
        // FloatC edge_lengths = zero<FloatC>();
        // set_slices(edge_lengths,slices(m_sec_edge_info.e1));
        // edge_lengths = select(edge_idx)
        // edge_lengths[edge_num] = norm(detach(m_sec_edge_info.e1[edge_num]));
        IntC num(edge_num);
        set_slices(num,slices(m_sec_edge_info.e1));
        FloatC ones(1.0f);
        set_slices(ones,slices(m_sec_edge_info.e1));
        
        FloatC edge_lengths = zero<FloatC>(slices(m_sec_edge_info.e1));
        IntC edge_idx = arange<IntC>(slices(edge_lengths));
        // slice(edge_lengths,edge_num) = ones;
        // edge_lengths.slice(edge_num) = 1.0f;
        // std::cout << "num " << num << std::endl;
         edge_lengths = select(edge_idx < 1,ones,0.0f);
        // std::cout << "edge_idpx " << edge_idx << std::endl;
        // std::cout << "edge_num " << edge_num << std::endl;

        // std::cout << "count(edge_idx == num) " << count(edge_idx == num) << std::endl;
        // std::cout << "num " << num << std::endl;
        // std::cout << "ones " << ones << std::endl;
        std::cout << "edge_lengths " << edge_lengths << std::endl;
        std::cout << "sum(edge_lengths) " << hsum(edge_lengths) << std::endl;
        std::cout << "count(edge_lengths != 0.0f) " << count(edge_lengths != 0.0f) << std::endl;

        std::cout << "Reset edge distribution to only sample  " << edge_num << std::endl;
        m_sec_edge_distrb->init(edge_lengths);
    }

    template <bool ad, bool path_space = false>
    Intersection<ad> ray_intersect(const Ray<ad> &ray, Mask<ad> active = true, TriangleInfoD *out_info = nullptr) const;

    template <bool ad, bool path_space = false>
    Intersection<ad> ray_all_intersect(const Ray<ad> &ray, Mask<ad> active, const Vector8f<ad> &sample, const int depth, TriangleInfoD *out_info = nullptr) const;

    template <bool ad, bool path_space = false>
    LayerIntersection ray_all_layer(const Ray<ad> &ray, Mask<ad> active, const int depth) const;

    template <bool ad>
    Spectrum<ad> Lenv(const Vector3f<ad> &wi, Mask<ad> active = true) const;

    template <bool ad>
    PositionSample<ad> sample_emitter_position(const Vector3f<ad> &ref_p, const Vector2f<ad> &sample, Mask<ad> active = true) const;

    template <bool ad>
    Float<ad> emitter_position_pdf(const Vector3f<ad> &ref_p, const Intersection<ad> &its, Mask<ad> active = true) const;

    BoundarySegSampleDirect sample_boundary_segment_direct(const Vector3fC &sample3, MaskC active = true) const;
    std::tuple<BoundarySegSampleDirect,SecondaryEdgeInfo> sample_boundary_segment_v2(const Vector3fC &sample3, MaskC active = true) const;
    std::tuple<BoundarySegSampleDirect,SecondaryEdgeInfo> sample_boundary_segment_v3(const Vector3fC &sample3, MaskC active = true, int edge_num =0) const;
    
    std::string to_string() const override;

    int                     m_num_sensors;
    int                     m_seed = 0;
    std::vector<Sensor*>    m_sensors;
    
    // Joon added: Used to compute silhouette edges
    std::vector<Sensor*>    m_emitter_sensors;

    std::vector<Emitter*>   m_emitters;
    EnvironmentMap          *m_emitter_env;
    EmitterArrayD           m_emitters_cuda;
    DiscreteDistribution    *m_emitters_distrb;
    DiscreteDistribution    *m_gradient_distrb; // domain: pixel. models intensity of gradient

    std::vector<BSDF*>      m_bsdfs;

    int                     m_num_meshes;
    std::vector<Mesh*>      m_meshes;
    MeshArrayD              m_meshes_cuda;

    // Scene bounding box
    Vector3fC               m_lower, m_upper;

    ParamMap                m_param_map;

    RenderOption            m_opts;
    mutable Sampler         *m_samplers;

protected:
    TriangleInfoD           m_triangle_info;
    TriangleUVD             m_triangle_uv;
    MaskD                   m_triangle_face_normals;
    bool                    m_has_bound_mesh;

    SecondaryEdgeInfo       m_sec_edge_info;
    DiscreteDistribution    *m_sec_edge_distrb;

    bool                    m_loaded;
    Scene_OptiX             *m_optix;

PSDR_CLASS_DECL_END(Scene)

} // namespace psdr
