// Copyright (c) 2013, Sandia Corporation.
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
// 
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
// 
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
// 
//     * Neither the name of Sandia Corporation nor the names of its
//       contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// 
#ifndef _BulkDataTester_hpp_
#define _BulkDataTester_hpp_

#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>   // for BulkData, etc
#include <stk_mesh/base/FEMHelpers.hpp>   // for BulkData, etc
#include <stk_mesh/baseImpl/MeshImplUtils.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/Types.hpp>      // for MeshIndex, EntityRank, etc
#include <stk_mesh/baseImpl/BucketRepository.hpp>  // for BucketRepository
#include <stk_mesh/base/EntityLess.hpp>
#include <stk_mesh/base/ElemElemGraph.hpp>

#include "BucketTester.hpp"

namespace stk { namespace mesh { namespace unit_test {

inline int does_entity_exist_in_list(const std::vector<stk::mesh::shared_entity_type>& shared_entities_this_proc, const stk::mesh::shared_entity_type &shared_entity_from_other_proc)
{
    int matching_index = -1;
    stk::topology topo2 = shared_entity_from_other_proc.topology;
    size_t num_nodes2 = shared_entity_from_other_proc.nodes.size();
    for (size_t i=0;i<shared_entities_this_proc.size();++i)
    {
        stk::topology topo1 = shared_entities_this_proc[i].topology;
        size_t num_nodes1 = shared_entities_this_proc[i].nodes.size();
        if (topo1 == topo2 && num_nodes1 == num_nodes2)
        {
            bool sameType = topo1.equivalent(shared_entities_this_proc[i].nodes, shared_entity_from_other_proc.nodes).first;
            if (sameType)
            {
                matching_index = i;
                break;
            }
        }
    }
    return matching_index;
}

class BulkDataTester : public stk::mesh::BulkData
{
public:

    BulkDataTester(stk::mesh::MetaData &mesh_meta_data, MPI_Comm comm) :
            stk::mesh::BulkData(mesh_meta_data, comm)
    {
    }

    BulkDataTester(stk::mesh::MetaData &mesh_meta_data, MPI_Comm comm, enum stk::mesh::BulkData::AutomaticAuraOption auto_aura_option) :
            stk::mesh::BulkData(mesh_meta_data, comm, auto_aura_option)
    {
    }

    BulkDataTester(stk::mesh::MetaData &mesh_meta_data, MPI_Comm comm, stk::mesh::ConnectivityMap const &conn_map) :
            stk::mesh::BulkData(mesh_meta_data, comm, stk::mesh::BulkData::AUTO_AURA, false, &conn_map)
    {
    }

    BulkDataTester(stk::mesh::MetaData &mesh_meta_data,
                   MPI_Comm comm,
                   enum stk::mesh::BulkData::AutomaticAuraOption auto_aura_option,
                   bool add_fmwk_data,
                   ConnectivityMap const* arg_connectivity_map,
                   FieldDataManager *field_data_manager,
                   unsigned bucket_capacity) :
            stk::mesh::BulkData(mesh_meta_data, comm, auto_aura_option, add_fmwk_data, arg_connectivity_map, field_data_manager, bucket_capacity)
    {
    }

    virtual ~BulkDataTester()
    {
    }

    bool is_entity_in_ghosting_comm_map(stk::mesh::Entity entity);

    void check_sharing_comm_maps();

    uint16_t closure_count(stk::mesh::Entity entity)
    {
        return m_closure_count[entity.local_offset()];
    }

    void reset_closure_count(stk::mesh::Entity entity)
    {
        m_closure_count[entity.local_offset()] = 0;
    }

    uint16_t my_orphaned_node_marking()
    {
        return orphaned_node_marking;
    }

    const stk::mesh::EntityCommDatabase my_entity_comm_map() const
    {
        return m_entity_comm_map;
    }

    bool is_ghosted_somewhere(stk::mesh::EntityKey key) const
    {
        return !internal_entity_comm_map(key, aura_ghosting()).empty();
    }

    void my_internal_change_entity_owner( const std::vector<stk::mesh::EntityProc> & arg_change, bool regenerate_aura = true, stk::mesh::impl::MeshModification::modification_optimization mod_optimization = stk::mesh::impl::MeshModification::MOD_END_SORT )
    {
        this->internal_change_entity_owner(arg_change,mod_optimization);
    }

    void my_resolve_ownership_of_modified_entities(const std::vector<stk::mesh::Entity> &shared_new)
    {
        this->resolve_ownership_of_modified_entities(shared_new);
    }

    bool my_entity_comm_map_insert(stk::mesh::Entity entity, const stk::mesh::EntityCommInfo & val)
    {
        return BulkData::entity_comm_map_insert(entity, val);
    }

    bool my_entity_comm_map_erase(const stk::mesh::EntityKey& key, const stk::mesh::EntityCommInfo& commInfo)
    {
        return BulkData::entity_comm_map_erase(key, commInfo);
    }

    bool my_entity_comm_map_erase(const stk::mesh::EntityKey& key, const stk::mesh::Ghosting& ghost)
    {
        return BulkData::entity_comm_map_erase(key, ghost);
    }

    void my_entity_comm_map_clear(const stk::mesh::EntityKey& key)
    {
        BulkData::entity_comm_map_clear(key);
    }

    void my_entity_comm_map_clear_ghosting(const stk::mesh::EntityKey& key)
    {
        BulkData::entity_comm_map_clear_ghosting(key);
    }

    bool my_internal_modification_end_for_change_entity_owner(stk::mesh::impl::MeshModification::modification_optimization opt )
    {
        return this->internal_modification_end_for_change_entity_owner(opt);
    }

    bool my_modification_end_for_entity_creation( EntityRank entity_rank, stk::mesh::impl::MeshModification::modification_optimization opt = stk::mesh::impl::MeshModification::MOD_END_SORT)
    {
        std::vector<EntityRank> entity_rank_vector = {entity_rank};
        return this->modification_end_for_entity_creation(entity_rank_vector, opt);
    }

    bool my_is_entity_in_sharing_comm_map(stk::mesh::Entity entity)
    {
        return this->is_entity_in_sharing_comm_map(entity);
    }

    void my_update_sharing_after_change_entity_owner()
    {
        this->update_sharing_after_change_entity_owner();
    }

    stk::mesh::impl::EntityRepository &my_get_entity_repository()
    {
        return get_entity_repository();
    }

    inline bool my_set_parallel_owner_rank_but_not_comm_lists(stk::mesh::Entity entity, int in_owner_rank)
    {
        return this->internal_set_parallel_owner_rank_but_not_comm_lists(entity, in_owner_rank);
    }

    bool my_internal_set_parallel_owner_rank_but_not_comm_lists(stk::mesh::Entity entity, int in_owner_rank)
    {
        return this->internal_set_parallel_owner_rank_but_not_comm_lists(entity, in_owner_rank);
    }

    void my_fix_up_ownership(stk::mesh::Entity entity, int new_owner)
    {
        this->fix_up_ownership(entity, new_owner);
    }

    stk::mesh::PairIterEntityComm my_internal_entity_comm_map_shared(const stk::mesh::EntityKey & key) const
    {
        return internal_entity_comm_map_shared(key);
    }

    int my_internal_entity_comm_map_owner(const EntityKey & key) const
    {
        return internal_entity_comm_map_owner(key);
    }

    const EntityCommListInfoVector & my_internal_comm_list() const
    {
        return internal_comm_list();
    }

    PairIterEntityComm my_internal_entity_comm_map(const EntityKey & key) const
    {
        return internal_entity_comm_map(key);
    }

    PairIterEntityComm my_internal_entity_comm_map(const EntityKey & key, const Ghosting & sub) const
    {
        return internal_entity_comm_map(key, sub);
    }

    bool my_comm_mesh_verify_parallel_consistency(std::ostream & error_log)
    {
        return comm_mesh_verify_parallel_consistency(error_log);
    }

    void my_internal_resolve_shared_modify_delete()
    {
        this->internal_resolve_shared_modify_delete();
    }

    void my_internal_resolve_ghosted_modify_delete()
    {
        this->internal_resolve_ghosted_modify_delete();
    }

    void my_internal_resolve_parallel_create()
    {
        this->internal_resolve_parallel_create();
    }

    void my_update_comm_list_based_on_changes_in_comm_map()
    {
        this->update_comm_list_based_on_changes_in_comm_map();
    }

    void my_internal_update_distributed_index(std::vector<stk::mesh::Entity> & shared_new )
    {
        this->internal_update_sharing_comm_map_and_fill_list_modified_shared_entities( shared_new );
    }

    void my_internal_update_distributed_index(stk::mesh::EntityRank entityRank, std::vector<stk::mesh::Entity> & shared_new )
    {
        this->internal_update_sharing_comm_map_and_fill_list_modified_shared_entities_of_rank(entityRank, shared_new);
    }

    void my_move_entities_to_proper_part_ownership( std::vector<stk::mesh::Entity> &shared_modified )
    {
        this->move_entities_to_proper_part_ownership( shared_modified );
    }

    void my_add_comm_list_entries_for_entities( std::vector<stk::mesh::Entity> &shared_modified )
    {
        this->add_comm_list_entries_for_entities( shared_modified );
    }

    void my_internal_resolve_shared_membership()
    {
        this->internal_resolve_shared_membership();
    }

    void my_internal_regenerate_aura()
    {
        this->internal_regenerate_aura();
    }

    void my_set_state(stk::mesh::Entity entity, stk::mesh::EntityState entity_state)
    {
        set_state(entity,entity_state);
    }

    void my_delete_shared_entities_which_are_no_longer_in_owned_closure()
    {
        delete_shared_entities_which_are_no_longer_in_owned_closure();
    }

    void my_ghost_entities_and_fields(Ghosting & ghosting, const std::set<EntityProc , EntityLess>& new_send)
    {
        ghost_entities_and_fields(ghosting, new_send);
    }

    void my_add_closure_entities(const stk::mesh::Ghosting& ghosting, const stk::mesh::EntityProcVec& entities, std::set <stk::mesh::EntityProc , stk::mesh::EntityLess > &entitiesWithClosure)
    {
        add_closure_entities(ghosting, entities, entitiesWithClosure);
    }

    void my_internal_modification_end_for_change_ghosting()
    {
        internal_modification_end_for_change_ghosting();
    }

    bool my_in_send_ghost(const stk::mesh::Ghosting& ghosting, stk::mesh::EntityKey key, int proc)
    {
        return in_send_ghost(ghosting, key, proc);
    }

    void my_markEntitiesForResolvingSharingInfoUsingNodes(stk::mesh::EntityRank entityRank, std::vector<stk::mesh::shared_entity_type>& shared_entities)
    {
        markEntitiesForResolvingSharingInfoUsingNodes(entityRank, shared_entities);
    }

    void my_fillSharedEntities(std::vector<shared_entity_type> & shared_entity_map, std::vector<std::vector<shared_entity_type> > &shared_entities)
    {
        fillVectorOfSharedEntitiesByProcessor(shared_entity_map, shared_entities);
    }

    void my_unpackEntityInfromFromOtherProcsAndMarkEntitiesAsSharedAndTrackProcessorsThatNeedAlsoHaveEntity(stk::CommSparse &comm,
            std::vector<stk::mesh::shared_entity_type> & shared_entity_map)
    {
        unpackEntityFromOtherProcAndUpdateInfoIfSharedLocally(comm, shared_entity_map);
    }

    void my_internal_change_entity_key(EntityKey old_key, EntityKey new_key, Entity entity)
    {
        internal_change_entity_key(old_key, new_key, entity);
    }

    stk::mesh::BulkData::entitySharing my_internal_is_entity_marked(stk::mesh::Entity entity) const
    {
        return internal_is_entity_marked(entity);
    }

    void my_set_mesh_index(Entity entity, Bucket * in_bucket, Bucket::size_type ordinal)
    {
        set_mesh_index(entity, in_bucket, ordinal);
    }

    stk::mesh::impl::BucketRepository& my_get_bucket_repository()
    {
        return get_bucket_repository();
    }
};

class BulkDataFaceSharingTester : public BulkDataTester
{
public:
    BulkDataFaceSharingTester(stk::mesh::MetaData &mesh_meta_data, MPI_Comm comm) :
            BulkDataTester(mesh_meta_data, comm)
    {
    }

    BulkDataFaceSharingTester(stk::mesh::MetaData &mesh_meta_data, MPI_Comm comm, enum stk::mesh::BulkData::AutomaticAuraOption auto_aura_option) :
        BulkDataTester(mesh_meta_data, comm, auto_aura_option)
    {
    }

    ~BulkDataFaceSharingTester(){}

    unsigned get_index_of_side_in_element_bucket(stk::mesh::Entity element, stk::mesh::Entity side);

    stk::mesh::Permutation get_permutation(stk::mesh::Entity element, const stk::mesh::EntityVector& nodes);

    stk::mesh::EntityVector convert_keys_to_entities(const std::vector<stk::mesh::EntityKey>& node_keys);

    void change_connectivity_for_edge_or_face(stk::mesh::Entity side, const std::vector<stk::mesh::EntityKey>& node_keys);

    void change_entity_key_and_nodes(const std::vector<shared_entity_type> & potentially_shared_sides);

    virtual void change_entity_key_and_update_sharing_info(std::vector<shared_entity_type> & potentially_shared_sides);

    void update_shared_entity_this_proc2(EntityKey global_key_other_proc, shared_entity_type& shared_entity, int proc_id, const std::vector<EntityKey>& nodes);

    virtual void check_if_entity_from_other_proc_exists_on_this_proc_and_update_info_if_shared(std::vector<stk::mesh::shared_entity_type>& shared_entities_this_proc,
            int proc_id, const stk::mesh::shared_entity_type &shared_entity_other_proc);

    virtual void sortNodesIfNeeded(std::vector<stk::mesh::EntityKey>& nodes) {}
};

class BulkDataElemGraphFaceSharingTester : public BulkDataFaceSharingTester
{
public:
    BulkDataElemGraphFaceSharingTester(stk::mesh::MetaData &mesh_meta_data, MPI_Comm comm) :
        BulkDataFaceSharingTester(mesh_meta_data, comm)
    {
    }

    BulkDataElemGraphFaceSharingTester(stk::mesh::MetaData &mesh_meta_data, MPI_Comm comm, enum stk::mesh::BulkData::AutomaticAuraOption auto_aura_option) :
        BulkDataFaceSharingTester(mesh_meta_data, comm, auto_aura_option)
    {
    }

    ~BulkDataElemGraphFaceSharingTester(){}

    virtual void markEntitiesForResolvingSharingInfoUsingNodes(stk::mesh::EntityRank entityRank, std::vector<shared_entity_type>& shared_entities);

    stk::mesh::Part& locally_owned_part() const { return mesh_meta_data().locally_owned_part(); }
    stk::mesh::EntityRank side_rank() const { return mesh_meta_data().side_rank(); }
    stk::mesh::EntityVector get_local_sides() const;
    bool entityWasCreatedThisModCycle(stk::mesh::Entity entity) const { return state(entity)==stk::mesh::Created; }
    stk::topology get_entity_topology(stk::mesh::Entity entity) const { return bucket(entity).topology(); }
    bool is_entity_owned(stk::mesh::Entity entity) const { return bucket(entity).owned(); }

protected:
    void determine_if_side_is_shared_across_proc_boundary_and_add(const stk::mesh::EntityVector& elements, stk::mesh::Entity side,
            const stk::mesh::ElemElemGraph& egraph, std::vector<shared_entity_type>& shared_entities);
    void use_elem_elem_graph_to_determine_shared_entities(std::vector<shared_entity_type>& shared_entities);
    void fill_shared_entities_that_need_fixing(const stk::mesh::EntityVector& sides, const stk::mesh::ElemElemGraph& egraph, std::vector<shared_entity_type>& shared_entities);
    void add_side_if_remote_element_connection_exists(int num_connected_elems, stk::mesh::Entity element, stk::mesh::Entity side, const stk::mesh::ElemElemGraph& egraph,
            std::vector<shared_entity_type>& shared_entities);
    void mark_entities_as_possibly_shared(const std::vector<shared_entity_type>& entities);
};

} } } // namespace stk mesh unit_test

#endif
