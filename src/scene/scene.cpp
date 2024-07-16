#include <chrono>
#include <misc/Exception.h>

#include <psdr/core/ray.h>
#include <psdr/core/intersection.h>
#include <psdr/core/sampler.h>
#include <psdr/core/pmf.h>
#include <psdr/bsdf/bsdf.h>
#include <psdr/emitter/envmap.h>
#include <psdr/sensor/perspective.h>
#include <psdr/shape/mesh.h>

#include <psdr/scene/scene_optix.h>
#include <psdr/scene/scene_loader.h>
#include <psdr/scene/scene.h>

namespace psdr
{

void LayerIntersection::reserve(int64_t s, int64_t d){
    size = s;
    depth = d;
    uvs = empty<Vector2fD>(s * d);
    shape_ids = empty<IntC>(s * d);
    numIntersections = empty<IntC>(s);
}

Scene::Scene() {
    m_has_bound_mesh = false;
    m_loaded = false;
    m_samplers = new Sampler[4];
    m_optix = new Scene_OptiX();
    m_emitter_env = nullptr;
    m_emitters_distrb = new DiscreteDistribution();
    m_gradient_distrb = new DiscreteDistribution();
    m_sec_edge_distrb = new DiscreteDistribution();
}


Scene::~Scene() {
    for ( Sensor*  s : m_sensors  ) delete s;
    for ( Emitter* e : m_emitters ) delete e;
    for ( BSDF*    b : m_bsdfs    ) delete b;
    for ( Mesh*    m : m_meshes   ) delete m;

    delete[]    m_samplers;
    delete      m_optix;
    delete      m_emitters_distrb;
    delete      m_gradient_distrb;
    delete      m_sec_edge_distrb;
}


void Scene::load_file(const char *file_name, bool auto_configure) {
    SceneLoader::load_from_file(file_name, *this);
    if ( auto_configure ) configure();
}


void Scene::load_string(const char *scene_xml, bool auto_configure) {
    SceneLoader::load_from_string(scene_xml, *this);
    if ( auto_configure ) configure();
}


void Scene::configure() {
    PSDR_ASSERT_MSG(m_loaded, "Scene not loaded yet!");
    PSDR_ASSERT(m_num_sensors == static_cast<int>(m_sensors.size()));
    PSDR_ASSERT(m_num_meshes == static_cast<int>(m_meshes.size()));

    using namespace std::chrono;
    auto start_time = high_resolution_clock::now();

    // Seed samplers
    std::cout<<"configure sampler"<<std::endl;
    if ( m_opts.spp  > 0 ) {
        int64_t sample_count = static_cast<int64_t>(m_opts.cropheight)*m_opts.cropwidth*m_opts.spp;
        std::cout << sample_count << std::endl;
        if ( m_samplers[0].m_sample_count != sample_count )
            m_samplers[0].seed(arange<UInt64C>(sample_count) + m_seed);
    }
    if ( m_opts.sppe > 0 ) {
        int64_t sample_count = static_cast<int64_t>(m_opts.cropheight)*m_opts.cropwidth*m_opts.sppe;
        if ( m_samplers[1].m_sample_count != sample_count )
            m_samplers[1].seed(arange<UInt64C>(sample_count) + m_seed);
    }
    if ( m_opts.sppse > 0 ) {
        // int64_t sample_count = static_cast<int64_t>(m_opts.sppse);
        // if ( m_samplers[2].m_sample_count != sample_count )
        //     m_samplers[2].seed(arange<UInt64C>(sample_count) + m_seed);
        // Joon added: tmp setting as m_samplers
        int64_t sample_count = static_cast<int64_t>(m_opts.cropheight)*m_opts.cropwidth*m_opts.sppse;
        std::cout << sample_count << std::endl;
        if ( m_samplers[2].m_sample_count != sample_count ){
            m_samplers[2].seed(arange<UInt64C>(sample_count) + m_seed);
        }
            
        // int64_t sample_count_2 = static_cast<int64_t>(m_opts.cropheight)*m_opts.cropheight*m_opts.sppsce;
        // if ( m_samplers[3].m_sample_count != sample_count_2)
        //     m_samplers[3].seed(arange<UInt64C>(sample_count_2));
    }

    // std::cout<<"configure meshes"<<std::endl;
    // Preprocess meshes
    PSDR_ASSERT_MSG(!m_meshes.empty(), "Missing meshes!");
    // std::cout<<"---------------number_of_meshes--------------------"<<(m_num_meshes)<<std::endl;
    std::vector<int> face_offset, edge_offset;
    face_offset.reserve(m_num_meshes + 1);
    face_offset.push_back(0);
    edge_offset.reserve(m_num_meshes + 1);
    edge_offset.push_back(0);
    m_lower = full<Vector3fC>(std::numeric_limits<float>::max());
    m_upper = full<Vector3fC>(std::numeric_limits<float>::min());
    int meshn = 0;
    for ( Mesh *mesh : m_meshes ) {
        // std::cout<<"configuring mesh id = "<<meshn<<std::endl;
        meshn = meshn+1;
        mesh->configure();
        // std::cout<<"configuring done for mesh "<<meshn<<std::endl;
        face_offset.push_back(face_offset.back() + mesh->m_num_faces);
        if ( m_opts.sppse > 0 && mesh->m_enable_edges )
            edge_offset.push_back(edge_offset.back() + static_cast<int>(slices(*(mesh->m_sec_edge_info))));
        else
            edge_offset.push_back(edge_offset.back());
        for ( int i = 0; i < 3; ++i ) {
            m_lower[i] = enoki::min(m_lower[i], hmin(detach(mesh->m_vertex_positions[i])));
            m_upper[i] = enoki::max(m_upper[i], hmax(detach(mesh->m_vertex_positions[i])));
        }
    }

    std::cout<<"configure sensor"<<std::endl;
    // Preprocess sensors
    PSDR_ASSERT_MSG(!m_sensors.empty(), "Missing sensor!");
    std::vector<size_t> num_edges;
    if ( m_opts.sppe > 0 ) num_edges.reserve(m_sensors.size());
    int sensorid = 0;
    for ( Sensor *sensor : m_sensors ) {
        // std::cout<<"configure sensor i = "<<sensorid<<std::endl;
        sensorid += 1;
        sensor->m_resolution = ScalarVector2i(m_opts.width, m_opts.height);
        sensor->m_cropsize   = ScalarVector2i(m_opts.cropwidth, m_opts.cropheight);
        sensor->m_cropoffset = ScalarVector2i(m_opts.cropoffset_x, m_opts.cropoffset_y);
        sensor->m_scene = this;
        sensor->configure();
        if ( m_opts.sppe > 0 ) num_edges.push_back(sensor->m_edge_distrb.m_size);

        for ( int i = 0; i < 3; ++i ) {
            if ( PerspectiveCamera *camera = dynamic_cast<PerspectiveCamera *>(sensor) ) {
                m_lower[i] = enoki::min(m_lower[i], detach(camera->m_camera_pos[i]));
                m_upper[i] = enoki::max(m_upper[i], detach(camera->m_camera_pos[i]));
            }
        }
    }
    if ( m_opts.log_level > 0 ) {
        std::stringstream oss;
        oss << "AABB: [lower = " << m_lower << ", upper = " << m_upper << "]";
        LOG(oss.str().c_str());

        if ( m_opts.sppe > 0 ) {
            std::stringstream oss;
            oss << "(" << num_edges[0];
            for ( size_t i = 1; i < num_edges.size(); ++i ) oss << ", " << num_edges[i];
            oss << ") primary edges initialized.";
            LOG(oss.str().c_str());
        }
    }

    // std::cout<<"configure lighting"<<std::endl;
    // Handling env. lighting
    if ( m_emitter_env != nullptr && !m_has_bound_mesh ) {
        FloatC margin = hmin((m_upper - m_lower)*0.05f);
        m_lower -= margin; m_upper += margin;

        m_emitter_env->m_lower = m_lower;
        m_emitter_env->m_upper = m_upper;

    // std::cout << "here " <<  std::endl;
        // Adding a bounding mesh
        std::array<std::vector<float>, 3> lower_, upper_;
        copy_cuda_array<float, 3>(m_lower, lower_);
        copy_cuda_array<float, 3>(m_upper, upper_);

        float vtx_data[3][8];
        for ( int i = 0; i < 8; ++i )
            for ( int j = 0; j < 3; ++j )
                vtx_data[j][i] = (i & (1 << j)) ? upper_[j][0] : lower_[j][0];

        const int face_data[3][12] = {
            0, 0, 1, 1, 2, 2, 0, 0, 0, 0, 4, 4,
            1, 3, 5, 7, 3, 7, 5, 4, 2, 6, 7, 6,
            3, 2, 7, 3, 7, 6, 1, 5, 6, 4, 5, 7
        };

        // std::cout << "here " <<  std::endl;
        Mesh *bound_mesh = new Mesh();
        bound_mesh->m_num_vertices = 8;
        bound_mesh->m_num_faces = 12;
        bound_mesh->m_use_face_normals = true;
        bound_mesh->m_enable_edges = false;
        bound_mesh->m_bsdf = nullptr;
        bound_mesh->m_emitter = m_emitter_env;
        bound_mesh->m_vertex_offset = zero<FloatD>(bound_mesh->m_num_vertices);
        for ( int i = 0; i < 3; ++i ) {
            bound_mesh->m_vertex_positions_raw[i] = FloatD::copy(vtx_data[i], 8);
            bound_mesh->m_face_indices[i] = IntD::copy(face_data[i], 12);
        }
        bound_mesh->configure();
        m_meshes.push_back(bound_mesh);

        face_offset.push_back(face_offset.back() + 12);
        edge_offset.push_back(edge_offset.back());

    // std::cout << "here " <<  std::endl;
        ++m_num_meshes;
        m_has_bound_mesh = true;
        if ( m_opts.log_level > 0 ) {
            LOG("Bounding mesh added for environmental lighting.");
        }
    }

    // Preprocess emitters
    if ( !m_emitters.empty() ) {
        std::vector<float> weights;
        weights.reserve(m_emitters.size());
        for ( Emitter *e : m_emitters ) {
            e->configure();
            weights.push_back(e->m_sampling_weight);
        }
        m_emitters_distrb->init(FloatC::copy(weights.data(), weights.size()));

        float inv_total_weight = rcp(m_emitters_distrb->m_sum)[0];
        for ( Emitter *e : m_emitters ) {
            e->m_sampling_weight *= inv_total_weight;
        }
    }

    // Initialize CUDA arrays
    if ( !m_emitters.empty() ) {
        m_emitters_cuda = EmitterArrayD::copy(m_emitters.data(), m_emitters.size());
    }
    m_meshes_cuda = MeshArrayD::copy(m_meshes.data(), m_num_meshes);

    // Generate global triangle arrays
    m_triangle_info = empty<TriangleInfoD>(face_offset.back());
    m_triangle_uv = zero<TriangleUVD>(face_offset.back());
    m_triangle_face_normals = empty<MaskD>(face_offset.back());
    for ( int i = 0; i < m_num_meshes; ++i ) {
        const Mesh &mesh = *m_meshes[i];
        const IntD idx = arange<IntD>(mesh.m_num_faces) + face_offset[i];
        
        scatter(m_triangle_info, *mesh.m_triangle_info, idx);
        scatter(m_triangle_face_normals, MaskD(mesh.m_use_face_normals), idx);
        if ( mesh.m_has_uv ) {
            scatter(m_triangle_uv, *mesh.m_triangle_uv, idx);
        }
    }

    // m_sec_edge_info = empty<SecondaryEdgeInfo>();
    // m_sec_edge_distrb->init(zero<FloatC>(1)); //slices(m_sec_edge_info.e1)));
    
    // Generate global sec. edge arrays
    if ( m_opts.sppse > 0 ) {
        m_sec_edge_info = empty<SecondaryEdgeInfo>(edge_offset.back());
        for ( int i = 0; i < m_num_meshes; ++i ) {
            const Mesh &mesh = *m_meshes[i];
            if ( mesh.m_enable_edges ) {
                PSDR_ASSERT(mesh.m_sec_edge_info != nullptr);
                const int m = static_cast<int>(slices(*mesh.m_sec_edge_info));
                
                const IntD idx = arange<IntD>(m) + edge_offset[i];
                // Grammar: target / source / idx
                // Relocate m_sec_edge info from mesh to variable
                scatter(m_sec_edge_info, *mesh.m_sec_edge_info, idx);
            }
        }
        
#if 0
        FloatC angle = select(detach(m_sec_edge_info.is_boundary), Pi,
                              safe_acos(dot(detach(m_sec_edge_info.n0), detach(m_sec_edge_info.n1))));
        m_sec_edge_distrb->init(norm(detach(m_sec_edge_info.e1))*angle);
#else
        m_sec_edge_distrb->init(norm(detach(m_sec_edge_info.e1)));
#endif
        if ( m_opts.log_level > 0 ) {
            std::stringstream oss;
            oss << edge_offset.back() << " secondary edges initialized.";
            LOG(oss.str().c_str());
        }
    } else {
        m_sec_edge_info = empty<SecondaryEdgeInfo>();
    }

    FloatC grads = full<FloatC>(1.0f);
    set_slices(grads,m_opts.cropheight * m_opts.cropwidth);
    m_gradient_distrb->init(grads);

    // Initialize OptiX
    cuda_eval();
    m_optix->configure(m_meshes);
    
    // auto p1 = m_sec_edge_info.p0 + m_sec_edge_info.e1;
    // auto p0 = m_sec_edge_info.p0;
    // const bool ad = false;
    // MaskC active(1.0f);
    // set_slices(active,slices(p0));
    // Vector2fC samples(1.0f);
    // set_slices(samples,slices(p0));
    // PositionSampleC ps2 = sample_emitter_position<false>(detach(p0), samples, active);
    // Vector3fC lightpoint = ps2.p;
    // set_slices(lightpoint, slices(p0)); 

    // // Vector3fC lightpoint = (m_emitters[0]->sample_position(Vector3fC(1.0f), Vector2fC(1.0f),active)).p;
    // // std::cout << "valid " << valid << std::endl;
    // // std::cout << "m_emitters[0]->m_position " << m_emitters[0]->m_position << std::endl;
    // std::cout << "lightpoint " << lightpoint << std::endl;
    
    // auto light2p0 = normalize(detach(p0) - lightpoint);
    // auto light2p1 = normalize(detach(p1)- lightpoint);

    // RayC rayP0 = RayC(lightpoint,light2p0);
    // RayC rayP1 = RayC(lightpoint,light2p1);
    // auto itsp0 = ray_intersect(rayP0);
    // auto itsp1 = ray_intersect(rayP1);
    // std::cout << "itsp0.p " << itsp0.p << std::endl;
    // std::cout << "itsp1.p " << itsp1.p << std::endl;
    // auto isShadowRay = itsp0.is_valid() ^ itsp1.is_valid();

    // std::cout << "isShadowRay " << isShadowRay << std::endl;
    // std::cout << "count(isShadowRay) " << count(isShadowRay) << std::endl;
    // auto edge_length_array = norm(detach(m_sec_edge_info.e1));
    // edge_length_array[~isShadowRay] = 0.0f;
    // std::cout << "Re-initialize edge length array " << std::endl;
    // m_sec_edge_distrb->init(edge_length_array);

    // Cleanup
    for ( int i = 0; i < m_num_meshes; ++i ) {
        Mesh *mesh = m_meshes[i];

        if ( mesh->m_emitter == nullptr ) {
            PSDR_ASSERT(mesh->m_triangle_info != nullptr);
            delete mesh->m_triangle_info;
            mesh->m_triangle_info = nullptr;

            if ( mesh->m_triangle_uv != nullptr ) {
                delete mesh->m_triangle_uv;
                mesh->m_triangle_uv = nullptr;
            }
        }

        if ( mesh->m_enable_edges ) {
            PSDR_ASSERT(mesh->m_sec_edge_info != nullptr);
            delete mesh->m_sec_edge_info;
            mesh->m_sec_edge_info = nullptr;
        }
    }

    auto end_time = high_resolution_clock::now();
    if ( m_opts.log_level > 0 ) {
        std::stringstream oss;
        oss << "Configured in " << duration_cast<duration<double>>(end_time - start_time).count() << " seconds.";
        LOG(oss.str().c_str());
    }
}


bool Scene::is_ready() const {
    return (m_opts.spp   == 0 || m_samplers[0].is_ready()) &&
           (m_opts.sppe  == 0 || m_samplers[1].is_ready()) &&
           (m_opts.sppse == 0 || (m_samplers[2].is_ready())) &&
           m_optix->is_ready();
}

/*Begin: xideng added for bssrdf */
template <bool ad, bool path_space>
Intersection<ad> Scene::ray_all_intersect(const Ray<ad> &ray, Mask<ad> active, const Vector8f<ad> &sample, const int depth, TriangleInfoD *out_info) const {
    static_assert(ad || !path_space);

    // std::cout << "before ray all intersect" << std::endl;
    Intersection<ad> its;
    Vector3i<ad> idx = m_optix->ray_all_intersect<ad>(ray, active, sample, depth);
    // std::cout << "after ray all intersect" << std::endl;
    // std::cout << "idx" << idx <<std::endl;

    TriangleInfo<ad>    tri_info;
    TriangleUV<ad>      tri_uv_info;
    Mask<ad>            face_normal_mask;
    if constexpr ( ad ) {
        tri_info = gather<TriangleInfoD>(m_triangle_info, idx[1], active);
        if ( out_info != nullptr ) *out_info = tri_info;

        tri_uv_info = gather<TriangleUVD>(m_triangle_uv, idx[1], active);
        face_normal_mask = gather<MaskD>(m_triangle_face_normals, idx[1], active);
        if constexpr ( path_space ) {
            its.J = tri_info.face_area/detach(tri_info.face_area);
        } else {
            its.J = 1.f;
        }
    } else {
        if ( out_info != nullptr ) {
            *out_info = gather<TriangleInfoD>(m_triangle_info, IntD(idx[1]), active);
            tri_info = detach(*out_info);
        } else {
            tri_info = gather<TriangleInfoC>(detach(m_triangle_info), idx[1], active);
        }

        tri_uv_info = gather<TriangleUVC>(detach(m_triangle_uv), idx[1], active);
        face_normal_mask = gather<MaskC>(detach(m_triangle_face_normals), idx[1], active);
        its.J = 1.f;
    }
    // std::cout << "tri_info.p0 " << tri_info.p0 << std::endl;
    its.n = tri_info.face_normal;

    const Vector3f<ad> &vertex0 = tri_info.p0, &edge1 = tri_info.e1, &edge2 = tri_info.e2;
    // std::cout << "vertex0 " << vertex0 << std::endl;

    if constexpr ( !ad || path_space ) {
        // Path-space formulation
        const Vector2fC &uv = m_optix->m_its.uv;

        //Vector3f<ad> sh_n = normalize(fmadd(tri_info.n0, 1.f - u - v, fmadd(tri_info.n1, u, tri_info.n2*v)));
        Vector3f<ad> sh_n = normalize(bilinear<ad>(tri_info.n0,
                                                   tri_info.n1 - tri_info.n0,
                                                   tri_info.n2 - tri_info.n0,
                                                   uv));
        masked(sh_n, face_normal_mask) = its.n;

        if constexpr ( ad ) {
            its.shape = gather<MeshArrayD>(m_meshes_cuda, idx[0], active);
        } else {
            its.shape = gather<MeshArrayC>(detach(m_meshes_cuda), idx[0], active);
        }
        its.p = bilinear<ad>(vertex0, edge1, edge2, uv);

        Vector3f<ad> dir = its.p - ray.o;
        its.t = norm(dir);
        dir /= its.t;

        its.sh_frame = Frame<ad>(sh_n);
        its.wi = its.sh_frame.to_local(-dir);
        //its.uv = fmadd(tri_uv_info[0], 1.f - u - v, fmadd(tri_uv_info[1], u, tri_uv_info[2]*v));
        its.uv = bilinear2<ad>(tri_uv_info[0],
                               tri_uv_info[1] - tri_uv_info[0],
                               tri_uv_info[2] - tri_uv_info[0],
                               uv);
        // Joon added 
        // Poly coeffs interpolation 
        for(int i=0;i<3;i++){
            its.poly_coeff[i] = bilinear20<ad>(tri_info.poly0[i],
                                            tri_info.poly1[i] - tri_info.poly0[i],
                                            tri_info.poly2[i] - tri_info.poly0[i],
                                            uv);
            // its.poly_coeff[i] = tri_info.poly0[i]; 

        }
        // std::cout << "poly_coeff " << poly_coeff << std::endl;
        // std::cout << "poly_coeff " << poly_coef << std::endl;
    } else {
        // Standard (solid-angle) formulation
        auto [uv, t] = ray_intersect_triangle<true>(vertex0, edge1, edge2, ray);

        //Vector3f<ad> sh_n = normalize(fmadd(tri_info.n0, 1.f - u_d - v_d, fmadd(tri_info.n1, u_d, tri_info.n2*v_d)));
        Vector3fD sh_n = normalize(bilinear<true>(tri_info.n0,
                                                  tri_info.n1 - tri_info.n0,
                                                  tri_info.n2 - tri_info.n0,
                                                  uv));
        masked(sh_n, face_normal_mask) = its.n;

        its.shape = gather<MeshArrayD>(m_meshes_cuda, idx[0], active);
        its.p = ray(t);
        its.t = t;

        its.sh_frame = Frame<ad>(sh_n);
        its.wi = its.sh_frame.to_local(-ray.d);
        //its.uv = fmadd(tri_uv_info[0], 1.f - u_d - v_d, fmadd(tri_uv_info[1], u_d, tri_uv_info[2]*v_d));
        its.uv = bilinear2<true>(tri_uv_info[0],
                                 tri_uv_info[1] - tri_uv_info[0],
                                 tri_uv_info[2] - tri_uv_info[0],
                                 uv);

        for(int i=0;i<3;i++){
            its.poly_coeff[i] = bilinear20<ad>(tri_info.poly0[i],
                                            tri_info.poly1[i] - tri_info.poly0[i],
                                            tri_info.poly2[i] - tri_info.poly0[i],
                                            uv);
            // its.poly_coeff[i] = tri_info.poly0[i]; 
            
        }
        // int num_samples = static_cast<int>(slices(ray.o));
        // FloatC tmp = zero<FloatC>(num_samples);
        // masked(tmp, detach(active)) = norm(Vector2fC(detach(u_d), detach(v_d)) - m_optix->m_its.uv);
        // std::cout << hmax(tmp) << std::endl;
    }
    

    its.num = idx[2];
    return its;
}
/*End: xideng added for bssrdf */

/*Begin: xideng added for multilayer bssrdf */
template <bool ad, bool path_space>
LayerIntersection Scene::ray_all_layer(const Ray<ad> &ray, Mask<ad> active, const int depth) const {
    static_assert(ad || !path_space);

    LayerIntersection layers;
    layers.reserve(static_cast<int>(slices(ray.o)), depth);
    Mask<ad> active1 = true;
    Vector2i<ad> idx = m_optix->ray_all_layer<ad>(ray, active1, depth);

    TriangleInfo<ad>    tri_info;
    TriangleUV<ad>      tri_uv_info;
    Mask<ad>            face_normal_mask;
    if constexpr ( ad ) {
        // std::cout<<"check 2"<<std::endl;
        tri_info = gather<TriangleInfoD>(m_triangle_info, idx[1], active1);
        tri_uv_info = gather<TriangleUVD>(m_triangle_uv, idx[1], active1);
        face_normal_mask = gather<MaskD>(m_triangle_face_normals, idx[1], active1);

    } else {
        tri_info = gather<TriangleInfoC>(detach(m_triangle_info), idx[1], active1);
        
        // std::cout<<"check 3"<<std::endl;
        tri_uv_info = gather<TriangleUVC>(detach(m_triangle_uv), idx[1], active1);
        
        face_normal_mask = gather<MaskC>(detach(m_triangle_face_normals), idx[1], active1);
    }

    const Vector3f<ad> &vertex0 = tri_info.p0, &edge1 = tri_info.e1, &edge2 = tri_info.e2;
    // std::cout<<"check 4"<<std::endl;
    if constexpr ( !ad || path_space ) {
        // Path-space formulation
        const Vector2fC &uv = m_optix->m_its.uvs;
       layers.uvs = bilinear2<ad>(tri_uv_info[0],
                               tri_uv_info[1] - tri_uv_info[0],
                               tri_uv_info[2] - tri_uv_info[0],
                               uv);
        // std::cout<<"computed"<<layers.uvs<<std::endl;
    } else {
        // Standard (solid-angle) formulation
        auto [uv, t] = ray_intersect_triangle<true>(vertex0, edge1, edge2, ray); // this ray has different size
       layers.uvs = bilinear2<true>(tri_uv_info[0],
                                 tri_uv_info[1] - tri_uv_info[0],
                                 tri_uv_info[2] - tri_uv_info[0],
                                 uv);

    }
    // return uv, depth, numofitersection, shape id
    
    
    layers.numIntersections = m_optix->m_its.numIntersections;
    layers.shape_ids = m_optix->m_its.shape_ids;

    return layers;
}
/*End: xideng added for multi-layer bssrdf */

template <bool ad, bool path_space>
Intersection<ad> Scene::ray_intersect(const Ray<ad> &ray, Mask<ad> active, TriangleInfoD *out_info) const {
    static_assert(ad || !path_space);

    Intersection<ad> its;
    // std::cout << active << std::endl;
    Vector2i<ad> idx = m_optix->ray_intersect<ad>(ray, active);
    // std::cout << "idx[1] " << idx[1] << std::endl;

    
    TriangleInfo<ad>    tri_info;
    TriangleUV<ad>      tri_uv_info;
    Mask<ad>            face_normal_mask;

    if constexpr ( ad ) {
        tri_info = gather<TriangleInfoD>(m_triangle_info, idx[1], active);
        if ( out_info != nullptr ) *out_info = tri_info;

        tri_uv_info = gather<TriangleUVD>(m_triangle_uv, idx[1], active);
        face_normal_mask = gather<MaskD>(m_triangle_face_normals, idx[1], active);
        if constexpr ( path_space ) {
            its.J = tri_info.face_area/detach(tri_info.face_area);
            // std::cout<<"triangle area"<<tri_info.face_area<<std::endl;
        } else {
            its.J = 1.f;
        }
    } else {
        if ( out_info != nullptr ) {
            *out_info = gather<TriangleInfoD>(m_triangle_info, IntD(idx[1]), active);
            tri_info = detach(*out_info);
        } else {
            tri_info = gather<TriangleInfoC>(detach(m_triangle_info), idx[1], active);
        }

        tri_uv_info = gather<TriangleUVC>(detach(m_triangle_uv), idx[1], active);
        face_normal_mask = gather<MaskC>(detach(m_triangle_face_normals), idx[1], active);
        its.J = 1.f;
    }
    its.n = tri_info.face_normal;

    const Vector3f<ad> &vertex0 = tri_info.p0, &edge1 = tri_info.e1, &edge2 = tri_info.e2;

    if constexpr ( !ad || path_space ) {
        // Path-space formulation
        const Vector2fC &uv = m_optix->m_its.uv;

        //Vector3f<ad> sh_n = normalize(fmadd(tri_info.n0, 1.f - u - v, fmadd(tri_info.n1, u, tri_info.n2*v)));
        Vector3f<ad> sh_n = normalize(bilinear<ad>(tri_info.n0,
                                                   tri_info.n1 - tri_info.n0,
                                                   tri_info.n2 - tri_info.n0,
                                                   uv));
        masked(sh_n, face_normal_mask) = its.n;

        if constexpr ( ad ) {
            its.shape = gather<MeshArrayD>(m_meshes_cuda, idx[0], active);
        } else {
            its.shape = gather<MeshArrayC>(detach(m_meshes_cuda), idx[0], active);
        }
        its.p = bilinear<ad>(vertex0, edge1, edge2, uv);

        Vector3f<ad> dir = its.p - ray.o;
        its.t = norm(dir);
        dir /= its.t;

        its.sh_frame = Frame<ad>(sh_n);
        // std::cout << "its.sh_frame " << its.sh_frame.n << std::endl;
        // std::cout << "sh_n " << sh_n << std::endl;
        its.wi = its.sh_frame.to_local(-dir);
        //its.uv = fmadd(tri_uv_info[0], 1.f - u - v, fmadd(tri_uv_info[1], u, tri_uv_info[2]*v));
        its.uv = bilinear2<ad>(tri_uv_info[0],
                               tri_uv_info[1] - tri_uv_info[0],
                               tri_uv_info[2] - tri_uv_info[0],
                               uv);
        // Joon added 
        // Poly coeffs interpolation 
        for(int i=0;i<3;i++){
            its.poly_coeff[i] = bilinear20<ad>(tri_info.poly0[i],
                                            tri_info.poly1[i] - tri_info.poly0[i],
                                            tri_info.poly2[i] - tri_info.poly0[i],
                                            uv);
            // its.poly_coeff[i] = tri_info.poly0[i]; 
        }
    } else {
        // Standard (solid-angle) formulation
        auto [uv, t] = ray_intersect_triangle<true>(vertex0, edge1, edge2, ray);

        //Vector3f<ad> sh_n = normalize(fmadd(tri_info.n0, 1.f - u_d - v_d, fmadd(tri_info.n1, u_d, tri_info.n2*v_d)));
        Vector3fD sh_n = normalize(bilinear<true>(tri_info.n0,
                                                  tri_info.n1 - tri_info.n0,
                                                  tri_info.n2 - tri_info.n0,
                                                  uv));
        masked(sh_n, face_normal_mask) = its.n;

        its.shape = gather<MeshArrayD>(m_meshes_cuda, idx[0], active);
        its.p = ray(t);
        its.t = t;

        its.sh_frame = Frame<ad>(sh_n);
        its.wi = its.sh_frame.to_local(-ray.d);
        //its.uv = fmadd(tri_uv_info[0], 1.f - u_d - v_d, fmadd(tri_uv_info[1], u_d, tri_uv_info[2]*v_d));
        its.uv = bilinear2<true>(tri_uv_info[0],
                                 tri_uv_info[1] - tri_uv_info[0],
                                 tri_uv_info[2] - tri_uv_info[0],
                                 uv);
        // Joon added 
        // Poly coeffs interpolation 
        for(int i=0;i<3;i++){
            its.poly_coeff[i] = bilinear20<ad>(tri_info.poly0[i],
                                            tri_info.poly1[i] - tri_info.poly0[i],
                                            tri_info.poly2[i] - tri_info.poly0[i],
                                            uv);
        }
        // int num_samples = static_cast<int>(slices(ray.o));
        // FloatC tmp = zero<FloatC>(num_samples);
        // masked(tmp, detach(active)) = norm(Vector2fC(detach(u_d), detach(v_d)) - m_optix->m_its.uv);
        // std::cout << hmax(tmp) << std::endl;
    }
    its.num = 1.0f;
    return its;
}


template <bool ad>
Spectrum<ad> Scene::Lenv(const Vector3f<ad> &wi, Mask<ad> active) const {
    return m_emitter_env == nullptr ? 0.f : m_emitter_env->eval_direction<ad>(wi, active);
}


template <typename T>
static void print_to_string(const std::vector<T*>& arr, const char* name, std::stringstream& oss) {
    if ( !arr.empty() ) {
        oss << "  # " << name << "\n";
        for ( size_t i = 0; i < arr.size(); ++i )
            oss << "  " << arr[i]->to_string() << "\n";
    }
}


template <typename T>
static void print_to_string(const Type<T*, true> &arr, const char *name, std::stringstream &oss) {
    if ( !arr.empty() ) {
        oss << "  # " << name << "\n";
        for ( size_t i = 0; i < slices(arr); ++i )
            oss << "  " << arr[i]->to_string() << "\n";
    }
}


std::string Scene::to_string() const {
    std::stringstream oss;
    oss << "Scene[\n";
    print_to_string<Sensor>(m_sensors, "Sensors", oss);
    oss << "\n";
    print_to_string<BSDF  >(m_bsdfs  , "BSDFs"  , oss);
    oss << "\n";
    print_to_string<Mesh  >(m_meshes , "Meshes" , oss);
    oss << "]";
    return oss.str();
}


template <bool ad>
PositionSample<ad> Scene::sample_emitter_position(const Vector3f<ad> &ref_p, const Vector2f<ad> &_sample2, Mask<ad> active) const {
    PSDR_ASSERT_MSG(!m_emitters.empty(), "No Emitter!");

    PositionSample<ad> result;
    if ( m_emitters.size() == 1U ) {
        result = m_emitters[0]->sample_position(ref_p, _sample2, active);
    } else {
        Vector2f<ad> sample2 = _sample2;
        auto [emitter_index, emitter_pdf] = m_emitters_distrb->sample_reuse<ad>(sample2.y());

        EmitterArray<ad> emitter_arr;
        if constexpr ( ad ) {
            emitter_arr = gather<EmitterArrayD>(m_emitters_cuda, IntD(emitter_index), active);
        } else {
            emitter_arr = gather<EmitterArrayC>(detach(m_emitters_cuda), emitter_index, active);
        }
        result = emitter_arr->sample_position(ref_p, sample2, active);
        result.pdf *= emitter_pdf;
    }
    return result;
}


template <bool ad>
Float<ad> Scene::emitter_position_pdf(const Vector3f<ad> &ref_p, const Intersection<ad> &its, Mask<ad> active) const {
    return its.shape->emitter(active)->sample_position_pdf(ref_p, its, active);
}

template <bool ad>
Vector3f<ad> interpolate(Vector3f<ad> p0, Vector3f<ad> p1){
    return (p0 + p1) / 2;
}
Vector3fD interpolate(Vector3fD p0, Vector3fD p1){
    return (p0 + p1) / 2;
}

template <bool ad>
Intersection<ad> Scene::bounding_edge(Scene scene, Intersection<ad> its, Vectorf<1,ad> r, Vector8f<ad> sample) const {
    
    auto phi = 2.0f * Pi * sample.x();
    auto vx = its.sh_frame.t;
    auto vy = its.sh_frame.s;
    auto vz = its.sh_frame.n;

    // Make bounding sphere
    // Float<ad> maxDist
    // TODO: set with variable
    // auto maxDist = 0.05f; 
    Float<ad> maxDist(0.05f);
    // set_slices(maxDist,slices(sample));

    // set_slices(vz,slices(sample));
    Ray<ad> ray(its.p + maxDist * (cos(phi)*vx + vy*sin(phi)),-vz,maxDist);
    Intersection<ad> its_curve;
    Mask<ad> active = its.p.x() != 0.0f;
    its_curve = this->ray_intersect<ad>(ray,active);
            // Intersection<ad> its3_back = scene->ray_intersect<ad, ad>(ray_back,active);
    // its_curve = ray_all_intersect<ad, ad>(ray, ray.o!=0.0f, sample, 5);
    IntC cameraIdx = arange<IntC>(slices(its_curve));
    IntC validRayIdx= cameraIdx.compress_(its_curve.is_valid());
    Intersection<ad> masked_its = gather<Intersection<ad>>(its_curve,Int<ad>(validRayIdx));

    return masked_its;
    // return its_curve;
    
}

template <bool ad>
Vector3fD Scene::crossing_edge(Vector3f<ad> _p,Vectorf<1,ad> sphereRadius, Vector2f<ad> sample) const {
    // Iterate through each edge
    // auto threshold = Vectorf<1,true>(sphereRadius);
    
    Vectorf<1,true> threshold = sphereRadius;
    Vector3fD p = _p; 
    if constexpr(!ad)
        {
        threshold = Vectorf<1,true>(sphereRadius);
        p = Vector3fD(_p);
        }
    // auto sensor = scene.m_sensors[sensor_id];
    
    // Create dynamic list of Boundary curve segments
    
    // For each triangle
    // Are there two crossing edges?
    // Connect the intersections
    auto p0 = m_triangle_info.p0;
    auto p1 = p0 + m_triangle_info.e1;
    auto p2 = p0 + m_triangle_info.e2;
    // std::cout << "p0 " << p0 << std::endl;
    // std::cout << "p1 " << p1 << std::endl;
    // std::cout << "p2 " << p2 << std::endl;
    
    // auto its_p = Vector3fD(p); //[N,3]
    // auto its_p = p;
    
    // List of curves!
    // TODO: which data structure should be use?
    // TODO: get the for loop out of my sight. Maybe use enoki::vectorize()
    // Fixed size? (dense) or dynamic arrays? (sparse)
    // Sample in advance?
    

    // TODO: vectorized version
    
    // For each intersection
    // Return a point on the boundary curve
    Vector3fD curve = enoki::zero<Vector3fD>(slices(p));
    std::cout << "slices(p) " << slices(p) << std::endl;
    
    for(int i=0;i<slices(p);i++){
        auto its_p = slice(p,i);
        
        auto dist0 = norm(p0-its_p);
        // std::cout << "blah" << std::endl;
        auto dist1 = norm(p1-its_p);
        auto dist2 = norm(p2-its_p);
        // std::cout << "blah" << std::endl;
        
        // Discarding indexing causes (I'm assuming) broadcasting errors
        auto d0 = dist0 < threshold[0];
        auto d1 = dist1 < threshold[0];
        auto d2 = dist2 < threshold[0]; 
        // std::cout << "blah" << std::endl;
        // std::cout << "d0 " << d0 << std::endl;
        // std::cout << "d1 " << d1 << std::endl;
        // std::cout << "d2 " << d2 << std::endl;
        
        auto valid = !(d0 && d1 && d2) && !(!d0 && !d1 && !d2);
        // std::cout << "blah" << std::endl;
        // std::cout << "count(valid) " << count(valid) << std::endl;
        
        // std::cout << "valid? " << std::endl;
        if (any(valid)){
            // TODO: fix
            set_slices(curve,count(valid)); //resize(curve,count(valid));
            // std::cout << "valid!" << std::endl;
            // std::cout << "count(valid) " << count(valid) << std::endl;
            Vector3fD cp0, cp1;
            IntC triIdx = arange<IntC>(slices(valid));
            auto validTriIdx = IntD(triIdx.compress_(valid));
            Vector3fD pp0 = gather<Vector3fD>(p0,validTriIdx);
            Vector3fD pp1 = gather<Vector3fD>(p1,validTriIdx);
            Vector3fD pp2 = gather<Vector3fD>(p2,validTriIdx);
            Vector3fD n = gather<Vector3fD>(m_triangle_info.face_normal,validTriIdx);
            auto dist00 = gather<FloatD>(dist0,validTriIdx);
            auto dist11 = gather<FloatD>(dist1,validTriIdx);
            auto dist22 = gather<FloatD>(dist2,validTriIdx);
            auto d00 = gather<MaskD>(d0,validTriIdx);
            auto d11 = gather<MaskD>(d1,validTriIdx);
            auto d22 = gather<MaskD>(d2,validTriIdx);
            // std::cout << "validTriIdx " << validTriIdx << std::endl;
            
            // TODO: change interpolation function
            cp0 = select(d00==d11,interpolate(pp0,pp2),interpolate(pp0,pp1));
            cp1 = select(d11==d22,interpolate(pp0,pp2),interpolate(pp1,pp2));

            // Connect cross points to get curve edge
            BoundaryCurveSampleDirect result;
            result.p0 = cp0;
            result.edge = cp1 - cp0;
            result.n = n;
            auto curveEdgeLength =norm(detach(result.edge));

            // Sample proportional to edge length
            DiscreteDistribution curve_distrb;
            curve_distrb.init(curveEdgeLength);

            // auto sample_single = FloatC(slice(sample.x(),i));
            // auto sample_single2 = FloatD(slice(sample.y(),i));
            // auto [edge_idx, pdf0] = curve_distrb.sample(sample_single); //sample.x());
            // Vector3fD curveP0 = gather<Vector3fD>(result.p0,IntD(edge_idx));
            // Vector3fD curvePedge = gather<Vector3fD>(result.edge,IntD(edge_idx));
            // Vector3fD curveP = curveP0 + sample_single2 * curvePedge;
            // scatter_add(curve, curveP, IntD(i));
            //
            auto sample_single = detach(sample.x()); //detach(sample.x());
            auto sample_single2 = sample.y();
            auto [edge_idx, pdf0] = curve_distrb.sample(sample_single);
                                                                        
            Vector3fD curveP0 = gather<Vector3fD>(result.p0,IntD(edge_idx));
            Vector3fD curvePedge = gather<Vector3fD>(result.edge,IntD(edge_idx));
            Vector3fD curveP = curveP0 + sample_single2 * curvePedge; // normalize(detach(curvePedge));
                                                                    
            // std::cout << "norm(curveP - p) " << norm(curveP - its_p) << std::endl;
                                                                
            // std::cout << "norm(curveP) " << norm(curveP) << std::endl;
            scatter_add(curve, curveP, arange<IntD>(slices(sample)));
            
        }
    }
        
    if constexpr(!ad){
        return detach(curve);
    }
    else{
        return curve;
    }
}
std::tuple<BoundarySegSampleDirect,SecondaryEdgeInfo> Scene::sample_boundary_segment_v3(const Vector3fC &sample3, MaskC active, int edge_num) const {
    // Sample a point p0 on a face edge
    BoundarySegSampleDirect result;

    FloatC sample1 = sample3.x();
    // auto [edge_idx, pdf0] = m_sec_edge_distrb->sample_reuse<false>(sample1);
    auto edge_length = norm(detach(m_sec_edge_info.e1));
    std::cout << "Edge length is " << slice(edge_length,edge_num) <<std::endl;
    active &= slice(edge_length, edge_num) >= 0.005f;
    std::cout << "ACTIVE"  <<active <<std::endl;
    
    IntC edge_idx(edge_num);
    FloatC pdf0(1.0f);
    set_slices(edge_idx,slices(sample1));
    set_slices(pdf0,slices(sample1));
    std::cout << "edge_idx " << edge_idx << std::endl;
    std::cout << "slices(m_sec_edge_info.e1) " << slices(m_sec_edge_info.e1) << std::endl;

    SecondaryEdgeInfo info = gather<SecondaryEdgeInfo>(m_sec_edge_info, IntD(edge_idx), active);
    result.p0 = fmadd(info.e1, sample1, info.p0); // p0 = info.p0 + info.e1*sample1;
    result.edge = normalize(detach(info.e1));
    result.edge2 = detach(info.p2) - detach(info.p0);
    const Vector3fC &p0 = detach(result.p0);
    pdf0 /= norm(detach(info.e1));
    // std::cout << "result.p0 " << result.p0.x() << std::endl;

    // Sample a point ps2 on a emitter

    PositionSampleC ps2 = sample_emitter_position<false>(p0, tail<2>(sample3), active);
    result.p2 = ps2.p;
    result.n = ps2.n;

    // Construct the edge "ray" and check if it is valid

    Vector3fC e = result.p2 - p0;
    const FloatC distSqr = squared_norm(e);
    e /= safe_sqrt(distSqr);
    const FloatC cosTheta = dot(result.n, -e);

    // IntC sgn0 = sign<false>(dot(detach(info.n0), e), EdgeEpsilon),
    //      sgn1 = sign<false>(dot(detach(info.n1), e), EdgeEpsilon);
    // result.is_valid = active && (cosTheta > Epsilon) && (
    //     (detach(info.is_boundary) && neq(sgn0, 0)) || (~detach(info.is_boundary) && (sgn0*sgn1 < 0))
    // );
    // std::cout << "count(active) " << count(active) << std::endl;
    // std::cout << "count(cosTheta > Epsilon) " << count(cosTheta > Epsilon) << std::endl;
    // std::cout << "count(sgn0*sgn1 <0) " << count(sgn0*sgn1 <0) << std::endl;
    // Joon: modified to sample arbitrary edges.
    result.is_valid = active && (cosTheta > Epsilon);

    result.pdf = (pdf0) & result.is_valid; //ps2.pdf*(distSqr/cosTheta)
    return std::tie(result,info);
}


std::tuple<BoundarySegSampleDirect,SecondaryEdgeInfo> Scene::sample_boundary_segment_v2(const Vector3fC &sample3, MaskC active) const {
    // Sample a point p0 on a face edge
    BoundarySegSampleDirect result;

    FloatC sample1 = sample3.x();
    auto [edge_idx, pdf0] = m_sec_edge_distrb->sample_reuse<false>(sample1);
    // auto [edge_idx, pdf0] = m_sec_edge_distrb->sample(sample1);
    // std::cout << "edge_idx " << edge_idx << std::endl;
    // Create Rays from emitter to edge.p0 and edge.p1
    // Discard ones with consistent visibility to light

    SecondaryEdgeInfo info = gather<SecondaryEdgeInfo>(m_sec_edge_info, IntD(edge_idx), active);
    result.p0 = fmadd(info.e1, sample1, info.p0); // p0 = info.p0 + info.e1*sample
    result.edge = normalize(detach(info.e1));
    result.edge2 = detach(info.p2) - detach(info.p0);
    const Vector3fC &p0 = detach(result.p0);
    pdf0 /= norm(detach(info.e1));
    // std::cout << "result.p0 " << result.p0.x() << std::endl;
    // std::cout << "ssmallest info.e1" << norm(info.e1) <<std::endl;
    // std::cout << "count" << count(norm(info.e1)<= 0.0f) <<std::endl;
    // std::cout << "count" << hmax(norm(info.e1)) <<std::endl;
    // std::cout << "count" << hmin(norm(info.e1)) <<std::endl;
    // std::cout << "count" << hmax(sample1) <<std::endl;
    // std::cout << "count" << hmax(sample1) <<std::endl;

    

    // Sample a point ps2 on a emitter

    PositionSampleC ps2 = sample_emitter_position<false>(p0, tail<2>(sample3), active);
    result.p2 = ps2.p;
    result.n = ps2.n;

    // Construct the edge "ray" and check if it is valid

    Vector3fC e = result.p2 - p0;
    const FloatC distSqr = squared_norm(e);
    e /= safe_sqrt(distSqr);
    const FloatC cosTheta = dot(result.n, -e);

    // IntC sgn0 = sign<false>(dot(detach(info.n0), e), EdgeEpsilon),
    //      sgn1 = sign<false>(dot(detach(info.n1), e), EdgeEpsilon);
    // result.is_valid = (cosTheta > Epsilon) && (
    //     (detach(info.is_boundary) && neq(sgn0, 0)) || (~detach(info.is_boundary) && (sgn0*sgn1 < 0))
    // );
    // std::cout << "count(active) " << count(active) << std::endl;
    // std::cout << "count(cosTheta > Epsilon) " << count(cosTheta > Epsilon) << std::endl;
    // std::cout << "count(sgn0*sgn1 <0) " << count(sgn0*sgn1 <0) << std::endl;
    // Joon: modified to sample arbitrary edges.
    result.is_valid = active && (cosTheta > Epsilon);
    // std::cout << result.is_valid << std::endl;
    // result.is_valid = active && (cosTheta > Epsilon);

    result.pdf = (pdf0) & result.is_valid; //ps2.pdf*(distSqr/cosTheta)
    return std::tie(result,info);
}

// Sample shadow edges
BoundarySegSampleDirect Scene::sample_boundary_segment_direct(const Vector3fC &sample3, MaskC active) const {

    BoundarySegSampleDirect result;

    // Sample a point p0 on a face edge

    FloatC sample1 = sample3.x();
    auto [edge_idx, pdf0] = m_sec_edge_distrb->sample_reuse<false>(sample1);
    // Create Rays from emitter to edge.p0 and edge.p1
    // Discard ones with consistent visibility to light

    SecondaryEdgeInfo info = gather<SecondaryEdgeInfo>(m_sec_edge_info, IntD(edge_idx), active);
    result.p0 = fmadd(info.e1, sample1, info.p0); // p0 = info.p0 + info.e1*sample1;
    result.edge = normalize(detach(info.e1));
    result.edge2 = detach(info.p2) - detach(info.p0);
    const Vector3fC &p0 = detach(result.p0);
    pdf0 /= norm(detach(info.e1));
    // std::cout << "result.p0 " << result.p0.x() << std::endl;

    // Sample a point ps2 on a emitter

    PositionSampleC ps2 = sample_emitter_position<false>(p0, tail<2>(sample3), active);
    result.p2 = ps2.p;
    result.n = ps2.n;

    // Construct the edge "ray" and check if it is valid

    Vector3fC e = result.p2 - p0;
    const FloatC distSqr = squared_norm(e);
    e /= safe_sqrt(distSqr);
    const FloatC cosTheta = dot(result.n, -e);

    IntC sgn0 = sign<false>(dot(detach(info.n0), e), EdgeEpsilon),
         sgn1 = sign<false>(dot(detach(info.n1), e), EdgeEpsilon);
    result.is_valid = active && (cosTheta > Epsilon) && (
        (detach(info.is_boundary) && neq(sgn0, 0)) || (~detach(info.is_boundary) && (sgn0*sgn1 < 0))
    );

    result.pdf = (pdf0) & result.is_valid; //ps2.pdf*(distSqr/cosTheta)
    return result;
}

// Explicit instantiations
template IntersectionC Scene::ray_intersect<false, false>(const RayC&, MaskC, TriangleInfoD*) const;
template IntersectionD Scene::ray_intersect<true , false>(const RayD&, MaskD, TriangleInfoD*) const;
template IntersectionD Scene::ray_intersect<true , true >(const RayD&, MaskD, TriangleInfoD*) const;

template IntersectionC Scene::ray_all_intersect<false, false>(const RayC&, MaskC, const Vector8fC&, const int, TriangleInfoD*) const;
template IntersectionD Scene::ray_all_intersect<true , false>(const RayD&, MaskD, const Vector8fD&, const int, TriangleInfoD*) const;
template IntersectionD Scene::ray_all_intersect<true , true >(const RayD&, MaskD, const Vector8fD&, const int, TriangleInfoD*) const;

template LayerIntersection Scene::ray_all_layer<false, false>(const RayC&, MaskC, const int) const;
template LayerIntersection Scene::ray_all_layer<true , false>(const RayD&, MaskD, const int) const;
template LayerIntersection Scene::ray_all_layer<true , true >(const RayD&, MaskD, const int) const;

template SpectrumC Scene::Lenv<false>(const Vector3fC&, MaskC) const;
template SpectrumD Scene::Lenv<true >(const Vector3fD&, MaskD) const;

template PositionSampleC Scene::sample_emitter_position<false>(const Vector3fC&, const Vector2fC&, MaskC) const;
template PositionSampleD Scene::sample_emitter_position<true >(const Vector3fD&, const Vector2fD&, MaskD) const;

template FloatC Scene::emitter_position_pdf<false>(const Vector3fC&, const IntersectionC&, MaskC) const;
template FloatD Scene::emitter_position_pdf<true >(const Vector3fD&, const IntersectionD&, MaskD) const;

template Vector3fD Scene::crossing_edge<true>(Vector3fD p, Vectorf<1,true> r, Vector2f<true> sample) const;
template Vector3fD Scene::crossing_edge<false>(Vector3fC p, Vectorf<1,false> r, Vector2f<false> sample) const;

template IntersectionD Scene::bounding_edge<true>(Scene scene, IntersectionD p, Vectorf<1,true> r, Vector8f<true> sample) const;
template IntersectionC Scene::bounding_edge<false>(Scene scene, IntersectionC p, Vectorf<1,false> r, Vector8f<false> sample) const;


} // namespace psdr
