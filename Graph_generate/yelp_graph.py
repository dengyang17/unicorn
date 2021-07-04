
class YelpGraph(object):

    def __init__(self, dataset):
        self.G = dict()
        self._load_entities(dataset)
        self._load_knowledge(dataset)
        self._clean()

    def _load_entities(self, dataset):
        print('load entities...')
        num_nodes = 0
        data_relations, _, _ = dataset.get_relation()  # entity_relations, relation_name, link_entity_type
        entity_list = list(data_relations.keys())
        for entity in entity_list:
            self.G[entity] = {}
            entity_size = getattr(dataset, entity).value_len
            for eid in range(entity_size):
                entity_rela_list = data_relations[entity].keys()
                self.G[entity][eid] = {r: [] for r in entity_rela_list}
            num_nodes += entity_size
            print('load entity:{:s}  : Total {:d} nodes.'.format(entity, entity_size))
        print('ALL total {:d} nodes.'.format(num_nodes))
        print('===============END==============')

    def _load_knowledge(self, dataset):
        _, data_relations_name, link_entity_type = dataset.get_relation()  # entity_relations, relation_name, link_entity_type
        for relation in data_relations_name:
            print('Load knowledge {}...'.format(relation))
            data = getattr(dataset, relation).data
            num_edges = 0
            for he_id, te_ids in enumerate(data):  # head_entity_id , tail_entity_ids
                if len(te_ids) <= 0:
                    continue
                e_head_type = link_entity_type[relation][0]
                e_tail_type = link_entity_type[relation][1]
                for te_id in set(te_ids):
                    self._add_edge(e_head_type, he_id, relation, e_tail_type, te_id)
                    num_edges += 2
            print('Total {:d} {:s} edges.'.format(num_edges, relation))
        print('===============END==============')

    def _add_edge(self, etype1, eid1, relation, etype2, eid2):
        self.G[etype1][eid1][relation].append(eid2)
        self.G[etype2][eid2][relation].append(eid1)

    def _clean(self):
        print('Remove duplicates...')
        for etype in self.G:
            for eid in self.G[etype]:
                for r in self.G[etype][eid]:
                    data = self.G[etype][eid][r]
                    data = tuple(sorted(set(data)))
                    self.G[etype][eid][r] = data
