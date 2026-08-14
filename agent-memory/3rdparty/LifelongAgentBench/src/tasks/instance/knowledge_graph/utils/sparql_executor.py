from typing import List, Tuple, Any
from SPARQLWrapper import SPARQLWrapper, JSON
import urllib
from urllib.error import URLError

from src.typings import TaskEnvironmentException
from src.utils import SafeLogger


class SparqlExecutor:
    def __init__(self, url: str):
        self.sparql_wrapper = SPARQLWrapper(url)
        self.sparql_wrapper.setReturnFormat(JSON)

    def _query_endpoint(self, query: str) -> dict[str, Any]:
        self.sparql_wrapper.setQuery(query)
        try:
            results = self.sparql_wrapper.query().convert()
        except urllib.error.URLError as e:
            SafeLogger.error(
                f"Cannot get result for query: {query}. Check whether the endpoint is reachable."
            )
            raise TaskEnvironmentException(f"Query failed:\n{query}") from e
        assert isinstance(results, dict)
        return results

    def execute_query(self, query: str) -> List[str]:
        results = self._query_endpoint(query)
        rtn = []
        for result in results["results"]["bindings"]:
            assert len(result) == 1  # only select one variable
            for var in result:
                rtn.append(
                    result[var]["value"]
                    .replace("http://rdf.freebase.com/ns/", "")
                    .replace("-08:00", "")
                )
        return sorted(rtn)

    def execute_unary(self, _type: str) -> List[str]:
        query = f"""PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX : <http://rdf.freebase.com/ns/> 
        SELECT (?x0 AS ?value) WHERE {{
        SELECT DISTINCT ?x0  WHERE {{
        ?x0 :type.object.type :{_type}. 
        }}
        }}"""
        results = self._query_endpoint(query)
        rtn = []
        for result in results["results"]["bindings"]:
            rtn.append(
                result["value"]["value"].replace("http://rdf.freebase.com/ns/", "")
            )
        return rtn

    def execute_binary(self, relation: str) -> List[Tuple[str, str]]:
        query = f"""PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX : <http://rdf.freebase.com/ns/>
        SELECT DISTINCT ?x0 ?x1 WHERE {{
        ?x0 :{relation} ?x1.
        }}"""
        results = self._query_endpoint(query)
        rtn = []
        for result in results["results"]["bindings"]:
            rtn.append((result["x0"]["value"], result["x1"]["value"]))
        return rtn

    def is_intersection(
        self, derivation1: tuple[Any, ...], derivation2: tuple[Any, ...]
    ) -> bool:
        if len(derivation1[1]) > 3 or len(derivation2[1]) > 3:
            return False
        if len(derivation1) == 2:
            clause1 = f"{derivation1[0]} {' / '.join(derivation1[1])} ?x. \n"
        elif len(derivation1) == 3:
            clause1 = f"?y {' / '.join(derivation1[1])} ?x. \nFILTER (?y {derivation1[2]} {derivation1[0]}) . \n"
        else:
            raise ValueError()

        if len(derivation2) == 2:
            clause2 = f"{derivation2[0]} {' / '.join(derivation2[1])} ?x. \n"
        elif len(derivation2) == 3:
            clause2 = f"?y {' / '.join(derivation2[1])} ?x. \nFILTER (?y {derivation2[2]} {derivation2[0]}) . \n"
        else:
            raise ValueError()

        query = f"""PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX : <http://rdf.freebase.com/ns/> 
        ASK {{
        {clause1}
        {clause2}
        }}"""
        results = self._query_endpoint(query)
        rtn = results["boolean"]
        assert isinstance(rtn, bool)
        return rtn

    def entity_type_connected(self, entity: str, _type: str) -> bool:
        query = f"""PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX : <http://rdf.freebase.com/ns/>
        ASK {{
        :{entity}  !(<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>|:type.object.type) / :type.object.type :{_type}
        }}"""
        results = self._query_endpoint(query)
        rtn = results["boolean"]
        assert isinstance(rtn, bool)
        return rtn

    def entity_type_connected_2hop(self, entity: str, _type: str) -> bool:
        query = f"""PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX : <http://rdf.freebase.com/ns/>
        ASK {{
        :{entity}  !(<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>|:type.object.type) / 
        !(<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>|:type.object.type) / :type.object.type :{_type}
        }}"""
        results = self._query_endpoint(query)
        rtn = results["boolean"]
        assert isinstance(rtn, bool)
        return rtn

    def get_in_attributes(self, value: str) -> list[str]:
        in_attributes = set()
        query = f"""PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX : <http://rdf.freebase.com/ns/>
        SELECT (?x0 AS ?value) WHERE {{
        SELECT DISTINCT ?x0  WHERE {{
        ?x1 ?x0 {value}.
        FILTER regex(?x0, "http://rdf.freebase.com/ns/")
        }}
        }}"""
        results = self._query_endpoint(query)
        for result in results["results"]["bindings"]:
            in_attributes.add(
                result["value"]["value"].replace("http://rdf.freebase.com/ns/", "")
            )
        return sorted(list(in_attributes))

    def get_in_relations(self, entity: str) -> list[str]:
        in_relations = set()
        query = f"""PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX : <http://rdf.freebase.com/ns/> 
        SELECT (?x0 AS ?value) WHERE {{
        SELECT DISTINCT ?x0  WHERE {{
        ?x1 ?x0 :{entity}.
        FILTER regex(?x0, "http://rdf.freebase.com/ns/")
        }}
        }}"""
        results = self._query_endpoint(query)
        for result in results["results"]["bindings"]:
            in_relations.add(
                result["value"]["value"].replace("http://rdf.freebase.com/ns/", "")
            )
        return sorted(list(in_relations))

    def get_in_entities(self, entity: str, relation: str) -> list[str]:
        query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX : <http://rdf.freebase.com/ns/>
        SELECT (?x1 AS ?value) WHERE {{
        SELECT DISTINCT ?x1  WHERE {{
        ?x1 :{relation} :{entity}.
        FILTER regex(?x1, "http://rdf.freebase.com/ns/")
        }}
        }}"""
        results = self._query_endpoint(query)
        neighbors = set()
        for result in results["results"]["bindings"]:
            neighbors.add(
                result["value"]["value"].replace("http://rdf.freebase.com/ns/", "")
            )
        return sorted(list(neighbors))

    def get_out_relations(self, entity: str) -> list[str]:
        query = f"""PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX : <http://rdf.freebase.com/ns/>
        SELECT (?x0 AS ?value) WHERE {{
        SELECT DISTINCT ?x0  WHERE {{
        :{entity} ?x0 ?x1.
        FILTER regex(?x0, "http://rdf.freebase.com/ns/")
        }}
        }}"""
        results = self._query_endpoint(query)
        out_relations = set()
        for result in results["results"]["bindings"]:
            out_relations.add(
                result["value"]["value"].replace("http://rdf.freebase.com/ns/", "")
            )
        return sorted(list(out_relations))

    def get_out_entities(self, entity: str, relation: str) -> list[str]:
        neighbors = set()
        query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX : <http://rdf.freebase.com/ns/>
        SELECT (?x1 AS ?value) WHERE {{
        SELECT DISTINCT ?x1  WHERE {{
        :{entity} :{relation} ?x1.
        FILTER regex(?x1, "http://rdf.freebase.com/ns/")
        }}
        }}"""
        results = self._query_endpoint(query)
        for result in results["results"]["bindings"]:
            neighbors.add(
                result["value"]["value"].replace("http://rdf.freebase.com/ns/", "")
            )
        return sorted(list(neighbors))


def main() -> None:
    sparql_executor = SparqlExecutor("http://222.201.139.67:3001/sparql")
    query = """PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT DISTINCT ?x
    WHERE {
    FILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))
    ns:m.025tbxf ns:type.object.type ?x .
    }"""
    results = sparql_executor.execute_query(query)
    print(results)


if __name__ == "__main__":
    main()
