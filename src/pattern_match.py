from __future__ import annotations
from copy import deepcopy, copy
import re, math
from typing_extensions import Callable
from collections import Counter, defaultdict

from timeit import default_timer as timer

#import matplotlib.pyplot as plt 
#import networkx as nx 

#from treelib import Node, Tree

import sys

assoc_ops = ('sum', 'prod', 'tprod')
#assoc_ops = ()

def process_word(word:str, stack:list[ParsedNode]):
    word = word.strip()
    if word == '':
        return

    #print(stack)
    op = None
    if word[-1] in (',','(',')','[',']'):
        op = word[-1]
        word = word[0:-1].strip()
    
    new_node = None
    if word != '':
        new_node = ParsedNode(word)
        stack[-1].children.append(new_node)
    if op != None and op in '([':
        if new_node is None:
            new_node = ParsedNode('')
            stack[-1].children.append(new_node)
        stack.append(new_node)
    elif op != None and op in ')]':
        stack.pop()

def parse_str(string:str):
    out = ParsedNode('root')
    stack = [out]

    #print(stack,'\n')

    #wordslist = []
    words = re.split(r"(?<=[(),])", string)
    for w in words:
        #wordslist.append(w)
        process_word(w,stack)
    #print('wordslist:\n', wordslist)
    if len(out.children) == 1:
        return out.children[0]
    
    raise Exception("")
    return tuple(out.children)

def parse_file(filename:str):
    out = ParsedNode('root')
    stack = [out]

    #print(stack,'\n')

    #wordslist = []
    with open(filename, 'r') as fp:
        for line in fp:
            words = re.split(r"(?<=[(),])", line)
            for w in words:
                #wordslist.append(w)
                process_word(w,stack)
    #print('wordslist:\n', wordslist)
    if len(out.children) == 1:
        return out.children[0]
    return tuple(out.children)


class ParsedNode():
    def __init__(self, name:str, children:list[ParsedNode]=[], type:str='op'):
        self.name = str(name)
        self.type = type
        self.children = copy(children)

    def apply_rule(self, rule:Rule, env:StateEnv = None):
        pattern_list = self.locate_rule_match(rule)
        out_list:list[ParsedNode] = []
        if env == None:
            env = StateEnv()

        for target_pattern in pattern_list:
            #print(rule, target_pattern)

            new_out, actual_target_pattern = deepcopy((self, target_pattern))

            if target_pattern != actual_target_pattern:
                raise Exception("This shouldn't happen")

            actual_target_pattern[0].apply_rule_at(rule, actual_target_pattern[2], actual_target_pattern[1][0], env)
            
            simplify_ops(new_out)
            out_list.append(new_out)

        #print()
        return out_list
    
    def apply_rule_at(self, rule:Rule, variable_mappings:dict[str,ParsedNode], index_offset:int=0, env:StateEnv = None):
        loc_variable_mappings = variable_mappings.copy()
        for k, func in rule.derived_params.items():
            val = func(variable_mappings, env)
            if isinstance(val, ParsedNode):
                loc_variable_mappings[k] = val
            else:
                loc_variable_mappings[k] = ParsedNode(val)
        
        prepared_input = rule.input.copy()
        prepared_output = rule.output.copy()
        for vname in variable_mappings.keys():
            prepared_input.replace_all(ParsedNode(vname), ParsedNode(f")({vname}"))
        for vname,val in variable_mappings.items():
            prepared_input.replace_all(ParsedNode(f")({vname}"), val)
        
        for vname in loc_variable_mappings.keys():
            prepared_output.replace_all(ParsedNode(vname), ParsedNode(f")({vname}"))
        for vname,val in loc_variable_mappings.items():
            prepared_output.replace_all(ParsedNode(f")({vname}"), val)

        #print(prepared_input, prepared_output)
        self.replace_node(prepared_input, prepared_output, index_offset)
        pass

    def __match_template(self:ParsedNode, target:ParsedNode, 
                         params:dict[str, ParsedNode]={}, 
                         index_range:tuple[int,int]|None=None, 
                         must_match_whole=False) -> list[tuple[ParsedNode, tuple[int, int], dict[str, ParsedNode]]]:
        out_list:list[tuple[ParsedNode, tuple[int,int], dict[str, ParsedNode]]] = []
        loc_params = deepcopy(params)
        if index_range == None:
            index_range = (0,len(self.children))
        
        #print(f"match: {self}, {target}")
        if target.name in loc_params.keys():
            if target.children != []:
                raise Exception("Improper rule format")

            if loc_params[target.name] == None:
                if self.name in assoc_ops and not must_match_whole:
                    start = index_range[0]
                    if self.name in assoc_op_id.keys():
                        loc_loc_params = loc_params.copy()
                        loc_loc_params[target.name] = assoc_op_id[self.name].copy()
                        out_list.append((self, (start,start), loc_loc_params))
                    for end in range(start+1, index_range[1]):
                        loc_loc_params = loc_params.copy()
                        loc_loc_params[target.name] = ParsedNode(self.name, [self.children[n] for n in range(start, end)]).copy()
                        out_list.append((self, (start, end), loc_loc_params))
                    return out_list
                else:
                    loc_params[target.name] = self
                    return [(self, index_range, loc_params)]
            elif self == loc_params[target.name]:
                return [(self, index_range, loc_params)]
            elif loc_params[target.name] != None and self.name in assoc_ops and not must_match_whole:
                raise Exception("Case currently not supported (and shouldn't be needed anytime soon)")
            else:
                return []
        elif target.name in assoc_op_id.keys() and self.name != target.name:
            
            #print("???", self, "<-->", target)
            if len(target.children) != 2:
                return []
            if target.children[0].name not in loc_params.keys() and target.children[0].name != assoc_op_id[target.name]:
                return []
            if loc_params[target.children[0].name] != assoc_op_id[target.name] and loc_params[target.children[0].name] != None:
                return []
            if len(target.children[0].children) > 0:
                return []
            
            loc_params[target.children[0].name] = assoc_op_id[target.name]

            out_data = self.__match_template(target.children[1], loc_params, must_match_whole=True)

            return [(self, index_range, params) for _,_,params in out_data]
            
        elif self.name != target.name:
            return []
        elif self.name in assoc_ops:
            
            out_ptr_list = [(index_range[0],loc_params)]
            is_first = True
            for n in target.children:
                new_out_ptr_list = []

                loc_first = is_first
                is_first = False

                for i,ptr_params in out_ptr_list:
                    if i >= index_range[1]:
                        if i > index_range[1] or n.name not in ptr_params.keys():
                            continue
                    else:
                        m = self.children[i]
                    if n.name in ptr_params.keys():
                        if ptr_params[n.name] == None:
                            for j in range(i, index_range[1]+1):
                                loc_ptr_params = ptr_params.copy()
                                if j-i == 0:
                                    if self.name not in assoc_op_id.keys() or not loc_first:
                                        continue
                                    loc_ptr_params[n.name] = assoc_op_id[self.name].copy()
                                elif j-i == 1:
                                    loc_ptr_params[n.name] = self.children[i]
                                else:
                                    loc_ptr_params[n.name] = ParsedNode(self.name, [p for p in self.children[i:j]])
                                new_out_ptr_list.append((j,loc_ptr_params))
                            
                        elif ptr_params[n.name].name == self.name:
                            j = i+len(ptr_params[n.name].children)
                            if j > index_range[1]:
                                continue
                            if ParsedNode(self.name, [p for p in self.children[i:j]]) == ptr_params[n.name]:
                                new_out_ptr_list.append((j,ptr_params))
                            pass
                        elif self.name in assoc_op_id.keys() and ptr_params[n.name] == assoc_op_id[self.name]:
                            new_out_ptr_list.append((i,ptr_params))
                        else:
                            new_out_ptr_list.extend([(i+1, p) for _,_,p in m.__match_template(n, ptr_params, must_match_whole=True)])
                    elif n.name == m.name or n.name in assoc_op_id.keys():
                        new_out_ptr_list.extend([(i+1, p) for _,_,p in m.__match_template(n, ptr_params, must_match_whole=True)])
                    

                out_ptr_list = new_out_ptr_list
                if out_ptr_list == []:
                    break

            out_list = [(self, (index_range[0], i),p) for i,p in out_ptr_list if (i >= index_range[1] or not must_match_whole)]
        else:
            if len(self.children) != len(target.children):
                return []
            out_list = [(self, (0,len(self.children)), loc_params)]
            for i, (s, t) in enumerate(zip(self.children, target.children)):
                new_out_list:list[tuple[ParsedNode, int, dict[str, ParsedNode]]] = []
                for _, _, p in out_list:
                    new_out_list.extend([(self, (0,len(self.children)), p) for _,_,p in s.__match_template(t, p, must_match_whole=True)])
                #print("new_out_list", i, new_out_list)
                out_list = new_out_list
                if out_list == []:
                    break


        #print('outlist:', self, out_list)
        return out_list


    def locate_template(self, target:ParsedNode, params:list[str]=[], detach_root=True, index_range:None|tuple[int,int]=None):
        out_list:list[tuple[ParsedNode, tuple[int,int], dict[str, ParsedNode]]] = []
        
        params_dict = dict((k, None) for k in params)

        if self.name not in assoc_ops:
            curr_outs = self.__match_template(target, params_dict.copy(), index_range)
            out_list.extend(curr_outs)
        else:
            if index_range == None:
                index_range = (0,len(self.children))
            for i in range(index_range[0], index_range[1]+1):
                curr_outs = self.__match_template(target, params_dict.copy(), index_range=(i, index_range[1]), must_match_whole=False)
                out_list.extend(curr_outs)
                #print("irange:", (i,index_range[1]))
                #print(out_list, '\n')

        #print("out_list:", self, out_list, '\n')

        if detach_root:
            for c in self.children:
                out_list.extend(c.locate_template(target, params))

        return out_list

    def locate_rule_match(self, rule:Rule, detach_root=True, index_range:None|tuple[int,int]=None):
        out_list:list[tuple[ParsedNode, tuple[int,int], dict[str, ParsedNode]]] = []
        for match in self.locate_template(rule.input, rule.params, detach_root, index_range):
            try:
                if not rule.has_condition or rule.conditions(match[2]):
                    out_list.append(match)
            except Exception as e:
                sys.stderr.write(f"Error checking conditions of: {match}\n")
                raise e
        return out_list
            

    def replace_node(self, replace_target:ParsedNode|str, replace_with:ParsedNode, index_offset:int=0):
        if isinstance(replace_target, str):
            replace_target = ParsedNode(replace_target)
        if self == replace_target:
            self.name = replace_with.name
            self.children = deepcopy(replace_with.children)
            self.type = replace_with.type
        elif self.name == replace_target.name and self.name in assoc_ops:
            new_children = self.children[0:index_offset]

            l = index_offset
            marked = False
            while l < len(self.children):
                r = l+len(replace_target.children)
                if not marked and r <= len(self.children) and self.children[l:r] == replace_target.children:
                    new_children.append(replace_with.copy())
                    l = r
                    marked = True
                else:
                    new_children.append(self.children[l])
                    l += 1

            #if not marked:
                #raise Exception("Note: pattern not found", f"{self} | {replace_target} -> {replace_with}")
                #print("Note: pattern not found", f"{self} | {replace_target} -> {replace_with}")

            self.children = new_children
            
        simplify_ops(self)

    def replace_all(self, replace_target:ParsedNode|str, replace_with:ParsedNode, index_offset:int=0):
        if isinstance(replace_target, str):
            replace_target = ParsedNode(replace_target)
        if self == replace_target:
            self.name = replace_with.name
            self.children = deepcopy(replace_with.children)
            self.type = replace_with.type
        else:
            if self.name == replace_target.name and self.name in assoc_ops:
                new_children = []

                l = index_offset
                while l < len(self.children):
                    r = l+len(replace_target.children)
                    if r < len(self.children) and self.children[l:r] == replace_target.children:
                        new_children.append(replace_with.copy())
                        l = r
                    else:
                        new_children.append(self.children[l])
                        self.children[l].replace_all(replace_target, replace_with)
                        l += 1

                self.children = new_children

            else:       
                for c in self.children:
                    c.replace_all(replace_target, replace_with)
                
            simplify_ops(self)

    def __repr__(self):
        out = f"{self.name}"
        out = out.strip(' ()')
        if len(self.children) > 0:
            out += f"( {', '.join([str(c) for c in self.children])})"
        return out

    def __hash__(self):
        return hash((self.name, tuple(self.children), self.type))

    def __len__(self):
        return 1 + sum([len(x) for x in self.children])

    def __eq__(self, other:ParsedNode):
        simplify_ops(self)
        if isinstance(other, ParsedNode):
            simplify_ops(other)
            try:
                return complex(self.name) == complex(other.name)
            except ValueError:
                return (self.name, self.type) == (other.name, other.type) and self.children == other.children 
        return super.__eq__(self, other)

    def copy(self):
        return deepcopy(self)

class Rule():

    rule_id_counter = 0
    
    trivial = lambda x:True

    def __init__(self, input:ParsedNode, params:list[str], output:ParsedNode, conditions:Callable[[dict[str, ParsedNode]], bool]|None=None, derived_params:dict[str, Callable[[dict[str, ParsedNode]], ParsedNode]]|None=None):
        self.input = input
        self.params = params
        self.output = output
        self.id = Rule.rule_id_counter
        Rule.rule_id_counter += 1

        self.has_condition = conditions != None
        if self.has_condition:
            self.conditions = conditions
        else:
            self.conditions = Rule.trivial

        if derived_params != None:
            self.derived_params:dict[str, Callable[[dict[str, ParsedNode], StateEnv], ParsedNode]] = derived_params
        else:
            self.derived_params:dict[str, Callable[[dict[str, ParsedNode], StateEnv], ParsedNode]] = dict()

    def __str__(self):
        if self.has_condition:
            return str((self.input, self.params + list(self.derived_params.keys()), self.output, "<condition>"))
        else:
            return str((self.input, self.params + list(self.derived_params.keys()), self.output))
        
    def __repr__(self):
        return f"<Rule #{self.id}>"


def apply_rules_list_greedy(target_node:ParsedNode, rules_list:list[Rule], env:StateEnv = None):
    checked:set[ParsedNode] = set()
    prev_ops:list[tuple[Rule,ParsedNode]] = []
    out = target_node.copy()

    skip_count = 0

    #print("`"*debug_level, "greedy search: ", target_node)

    while skip_count < len(rules_list):
        items:set[tuple[Rule,ParsedNode]] = set()
        
        for rule in rules_list:
            items |= set((rule,x) for x in out.apply_rule(rule, env))
            if len(items) > 0:
                break

        items = [i for i in items if i[1] not in checked]

        if len(items) == 0:
            skip_count += 1
        else:
            skip_count = 0
            n, g = items.pop()

            prev_ops.append((n, g))
            out = g
            checked.add(out)

            #print("`"*debug_level, ">> item:", out)
            #print("`"*debug_level, ">> prev_ops", [r for r,_ in prev_ops])


        #for x in buffer:
        #    print(x, '|')
        #print()

    simplify_ops(out)
    return [(prev_ops, out)]

class StateEnv():
    def __init__(self):
        self.history:list[tuple[Rule,ParsedNode]] = []
        self.index_counters = Counter()
        self.rule_counter:Counter = None
        self.layer = 0

    def copy(self):
        out = StateEnv()
        out.history = self.history.copy()
        out.index_counters = copy(self.index_counters)
        out.rule_counter = self.rule_counter
        out.layer = self.layer + 1
        return out

debug_level = 0

def apply_rules_list_full_search(target_node:ParsedNode, recursive_rules_list:list[Rule],
                                 branching_rules_list:list[Rule],
                                 greedy_rules_list:list[Rule],
                                 end_condition=None, dead_end_patterns:list[tuple[ParsedNode, list[str], Callable]]=[],
                                 env:StateEnv = None):
    global debug_level
    all_branching_rules = branching_rules_list+recursive_rules_list
    debug_level += 1
    
    out_res:list[tuple[StateEnv,ParsedNode]] = []
    checked:set[ParsedNode] = set()
    rule_counts = Counter()
    if env == None:
        start_env = StateEnv()
    else:
        start_env = env


    #print("`"*start_env.layer, f"full search: {target_node}")


    buffer:list[tuple[StateEnv,ParsedNode]] = []
    prevops, reduced_node = apply_rules_list_greedy(target_node, greedy_rules_list)[0]
    start_env.history.extend(prevops)
    rule_counts.update((r[0].id for r in prevops))
    buffer.append((start_env, reduced_node))

    valid_result = None

    while len(buffer) > 0:
        loc_env, n = buffer.pop()
        #print("`"*loc_env.layer, f"> testing: {n}")
        branching_history = [i for i,_ in loc_env.history if i in [r.id for r in all_branching_rules]]
        #print("`"*loc_env.layer, f"  {[r.id for r in branching_history]}")

        early_term_flag = False
        for pat in dead_end_patterns:
            if len(pat) == 3:
                match, params, cond = pat
            else:
                match, params = pat
                cond = lambda x : True
            if any([cond(hit) for _,_,hit in n.locate_template(match, params)]):
                early_term_flag = True
                break

        if early_term_flag:
            out_res.append((loc_env, n))
            #print("`"*loc_env.layer, f"dead end: {(loc_env, n)}")
            continue

        last_prev_op = None
        if len(branching_history) > 0:
            last_prev_op = branching_history[-1]

        items:list[tuple[tuple[Rule,ParsedNode],tuple[ParsedNode, tuple[int, int], dict[str, ParsedNode]]]] = list()
        for rule in all_branching_rules:
            if rule == last_prev_op:
                continue
            items.extend([((rule,n), q) for q in n.locate_rule_match(rule)])
        
        if last_prev_op != None and last_prev_op in all_branching_rules:
            items.extend([((last_prev_op,n), q) for q in n.locate_rule_match(last_prev_op)])
        
        n_items:list[tuple[StateEnv,ParsedNode]] = []

        for prvrule, (target, index_range, vals) in items:
            out_portion = target.copy()
            out_portion.children = out_portion.children[index_range[0]:index_range[1]]

            loc_hist_old = len(loc_env.history)

            new_env = loc_env.copy()
            out_rule_applied = out_portion.copy()
            out_rule_applied.apply_rule_at(prvrule[0], vals, 0, new_env)
            #print("index_counters:", new_env.index_counters)
            new_env.history.append((prvrule[0],n))
            rule_counts[prvrule[0].id] += 1

            if end_condition != None and not end_condition(out_rule_applied) and prvrule in recursive_rules_list:
                
                old_val = len(new_env.history)
                out_final_state, out_states, loc_rule_count = apply_rules_list_full_search(out_rule_applied, recursive_rules_list, branching_rules_list, greedy_rules_list, end_condition, dead_end_patterns, new_env)
                rule_counts.update(loc_rule_count)

                for dead_end_state, exp in out_states:
                    if out_final_state != None and out_final_state[1] == exp:
                        continue
                    out_res.append((dead_end_state, exp))

                if out_final_state == None:
                    continue
                new_env = out_final_state[0]
                new_env.layer -= 1
                out_rule_applied = out_final_state[1]
            
            prv_hist, out_rule_applied = apply_rules_list_greedy(out_rule_applied, greedy_rules_list)[0]
            new_env.history += prv_hist
            rule_counts.update((r[0].id for r in prv_hist))

            rule_counts.update([r[0].id for r in prv_hist])
                

            new_base, new_target = deepcopy((n, target))
            new_target.replace_node(out_portion, out_rule_applied, index_range[0])
            n_items.append((new_env, new_base))


        full_size_len = len(n_items)
        rem_by_check = [i for i in n_items if i[1] in checked]
        #if len(rem_by_check) > 0:
        #    print("`"*loc_env.layer, "removed by check:", rem_by_check)
        n_items = [i for i in n_items if i[1] not in checked]
        #print("`"*loc_env.layer, "buffer shrink:", full_size_len, '->', len(n_items))
        

        #print("children index hits:", [i.history[-1][0].id for i,_ in n_items])
        
        if len(n_items) == 0:
            if full_size_len > 0:
                #print("`"*loc_env.layer, f"cut path: {(n)}")
                pass
            else:
                out_res.append((loc_env, n))

                if end_condition != None and end_condition(n):
                    #print("`"*loc_env.layer, f"result: {(n)}")
                    valid_result = (loc_env, n)
                    break
                else:
                    #print("`"*loc_env.layer, f"dead end: {(n)}")
                    pass
        else:
            buffer.extend(n_items)
            checked = checked.union([i for _,i in n_items])

            #print("`"*loc_env.layer, [i[0].history[-1][0] for i in buffer])


    debug_level -= 1
    return valid_result, out_res, rule_counts

assoc_op_id = dict([
    #('sum',ParsedNode('0'),),
    ('prod',ParsedNode('1'),),
])

def simplify_ops(node:ParsedNode):
    for c in node.children:
        simplify_ops(c)

    flag = True
    while flag:
        flag = False

        if node.name == '' and len(node.children) == 1:
            c = node.children[0]
            node.name = c.name
            node.children = c.children

        if node.name == 'mod_shift' and len(node.children) == 2:
            c1, c2 = node.children
            try:
                i1, i2, = int(c1.name), int(c2.name)
                node.children = []
                node.name = str((i1 + i2) % 3)
                if node.name == '0':
                    node.name = '3'
            except ValueError:
                pass

        
        if node.name == 'sqrt' and len(node.children) == 1:
            c1 = node.children[0]
            if len(c1.children) == 0:
                try:
                    val = float(c1.name)
                    out_val = math.sqrt(val)
                    node.name = str(out_val)
                    node.children = []
                except ValueError:
                    pass
        elif node.name == 'nsign' and len(node.children) == 1:
            c1 = node.children[0]
            if len(c1.children) == 0:
                try:
                    val = complex(c1.name)
                    out_val = -val
                    node.name = str(out_val)
                    node.children = []
                except ValueError:
                    pass
            elif c1.name == 'prod':
                c1.children = [parse_str("-1")] + c1.children
                node.name = "prod"
                node.children = c1.children
        elif node.name == 'conj' and len(node.children) == 1:
            c1 = node.children[0]
            if len(c1.children) == 0:
                try:
                    val = complex(c1.name)
                    out_val = complex.conjugate(val)
                    node.name = str(out_val)
                    node.children = []
                except ValueError:
                    pass


        if node.name in assoc_ops:
            new_children:list[ParsedNode] = []
            for c in node.children:
                if c.name == node.name:
                    new_children.extend(c.children)
                elif node.name in assoc_op_id.keys() and c == assoc_op_id[node.name]:
                    pass
                elif len(new_children) > 0 and _is_const(c) and _is_const(new_children[-1]):
                    out_c = complex(new_children.pop().name)
                    if node.name == 'prod':
                        new_children.append(ParsedNode(f"{out_c * complex(c.name)}"))
                    elif node.name == "sum":
                        new_children.append(ParsedNode(f"{out_c + complex(c.name)}"))
                    flag = True
                else:
                    new_children.append(c)
            node.children = new_children

        if len(node.children) == 1 and node.name in assoc_ops:
            node.name = node.children[0].name
            node.children = node.children[0].children

        elif len(node.children) == 0 and node.name in assoc_op_id.keys():
            node.name = assoc_op_id[node.name].name


def _is_const(node:ParsedNode):
    if node.name in ('nsign', 'sum', 'prod', 'complex'):
        return all(_is_const(c) for c in node.children)
    try:
        complex(node.name)
        return True
    except ValueError:
        return False

def __is_real_const(node:ParsedNode):
    if node.name in ('nsign', 'sum', 'prod'):
        return all(_is_const(c) for c in node.children)
    try:
        float(node.name)
        return True
    except ValueError:
        return False

def __is_squared_factor(node:ParsedNode|complex, unit_scalar:complex):
    try:
        if isinstance(node, ParsedNode):
            if node.children != []:
                return False
            z = complex(node.name)
        else:
            z = node
        z2 = z/unit_scalar
        if z2.imag != 0 or z2.real < 0:
            return False
        return True
    except ValueError:
        return False

def __get_squared_factor(node:ParsedNode|complex, unit_scalar:complex):
    if isinstance(node, ParsedNode):
        z = complex(node.name)
    else:
        z = node
    z2 = z/unit_scalar
    if z2.imag != 0:
        raise Exception(f"{z} -> {z2} cannot be squared")
    return math.sqrt(z2.real)

def __check_equality_list(vars:dict[str, ParsedNode], conditions_list:list[tuple[ParsedNode, ParsedNode]], reduction_rules:list[Rule]=None):
    prepared_conds = [(c1.copy(), c2.copy()) for c1,c2 in conditions_list]
    for vname,val in vars.items():
        for c1, c2 in prepared_conds:
            c1.replace_all(ParsedNode(vname), val)
            c2.replace_all(ParsedNode(vname), val)
    
    if reduction_rules == None:
        reduction_rules = basic_rules_list

    for c1, c2 in prepared_conds:
        #print(f"comparing: {c1}, {c2}")
        nc1 = [x[1] for x in apply_rules_list_greedy(c1, reduction_rules)]
        nc2 = [x[1] for x in apply_rules_list_greedy(c2, reduction_rules)]
        if any(o1 == o2 for o1 in nc1 for o2 in nc2):
            #print("compare passed")
            continue
        else:
            return False
    return True

def __increment_counter(env:dict, key:str):
    if key not in env.keys():
        env[key] = 0
    out = env[key]
    env[key] += 1
    return ParsedNode(str(out)) 

def __is_mode(node:ParsedNode):
    try:
        int(node.name)
        return False
    except ValueError:
        pass
    if node.name == '1' or node.name in assoc_ops:
        return False
    if node.children == []:
        return True
    if len(node.children) == 1:
        try:
            int(node.children[0].name)
            return True
        except ValueError:
            return False
    return False

terminate_patterns = [
    #1
    (parse_str('BOD(1)'),
     [],
    ),
    #(parse_str('prod(1)'),
    # [],
    #),
]


commute_rules_list = [
    Rule(parse_str('sum(prod(?c, ?1,?2),prod(?nc,?2,?1)))'), ['?1', '?2', '?c', '?nc'], parse_str('prod(?c, comm(?1, ?2))'),
         lambda x : not __check_equality_list(x, [(parse_str("?1"), parse_str("?2"))]) and
        __check_equality_list(x, [(parse_str("?c"), parse_str("nsign(?nc)"))])
         
         ),
    Rule(parse_str('sum(prod(?1,?2),prod(?2,?1))'), ['?1', '?2'], parse_str('acomm(?1, ?2)')),
    #(parse_str('comm(?1, ?1)'), ['?1'], parse_str('0')),
    #(parse_str('acomm(?1, ?1)'), ['?1'], parse_str('prod(2, ?1)')),
    #(parse_str('comm(?1, ?2)'), ['?1', '?2'], parse_str('sum(prod(?1,?2),nsign(prod(?2,?1)))')),
    #(parse_str('acomm(?1, ?2)'), ['?1', '?2'], parse_str('sum(prod(?1,?2),prod(?2,?1))')),
]

gate_cancel_rules = [
    Rule(parse_str('prod(qgate(h, qb1), qgate(h, qb1))'), ['qb1'], parse_str('prod')),
    Rule(parse_str('prod(qgate(x, qb1), qgate(x, qb1))'), ['qb1'], parse_str('prod')),
    Rule(parse_str('prod(qgate(s, qb1), qgate(dagger(s), qb1))'), ['qb1'], parse_str('prod')),
    Rule(parse_str('prod(qgate(dagger(s), qb1), qgate(s, qb1))'), ['qb1'], parse_str('prod')),
]

basic_rules_list = [
    Rule(parse_str('dagger(tprod(?1, ?2))'), ['?1', '?2'], parse_str('tprod(dagger(?2), dagger(?1)))'),
     lambda x : x['?1'].name != '1' and x['?2'].name != '1'),
    Rule(parse_str('dagger(prod(?t, ?1, ?2))'), ['?1', '?2', '?t'], parse_str('prod(?t, dagger(?2), dagger(?1)))'),
     lambda x : x['?1'].name != '1' and x['?2'].name != '1' and _is_const(x['?t'])),
    Rule(parse_str('dagger(prod(?1, ?2))'), ['?1', '?2'], parse_str('prod(dagger(?2), dagger(?1)))'),
     lambda x : x['?1'].name != '1' and x['?2'].name != '1'),
    Rule(parse_str('dagger(dagger(?1))'), ['?1'], parse_str('?1')),
    Rule(parse_str('prod(-1,-1,?1)'), ['?1'], parse_str('?1')),
    Rule(parse_str('sum(prod(?2,?1), prod(?3,?1))'), ['?1','?2','?3'], parse_str('0'),
         lambda x : __check_equality_list(x, [(parse_str('?2'), parse_str('nsign(?3)'))], [])  
         ),
    Rule(parse_str('sum(prod(?2,?1), prod(?3,?1))'), ['?1','?2','?3'], parse_str('prod(?4,?1)'),
         lambda x : _is_const(x['?2']) and _is_const(x['?3']) and
         not __check_equality_list(x, [(parse_str('?2'), parse_str('nsign(?3)'))], []) ,
         {'?4': (lambda x,_: complex(x['?2'].name) + complex(x['?3'].name))}
         ),
    #Rule(parse_str('prod(bc(?1), ba(?2))'), ['?1', '?2'], parse_str('1')),
    Rule(parse_str('square(?1)'), ['?1'], parse_str('prod(?1, ?1)')),
    Rule(parse_str('exp(0)'), [], parse_str('1')),

    #Rule(parse_str('conj(sum(?1, prod(j,?2)))'), ['?1', '?2'], parse_str('sum(?1, prod(-j,?2))'), lambda x: __is_real_const(x['?1']) and __is_real_const(x['?2'])),
    #Rule(parse_str('conj(sum(?1, nsign(prod(j,?2))))'), ['?1', '?2'], parse_str('sum(?1, prod(j,?2))'), lambda x: __is_real_const(x['?1']) and __is_real_const(x['?2'])),
    #Rule(parse_str('conj(sum(prod(j,?1), ?2))'), ['?1', '?2'], parse_str('sum(nsign(prod(j,?1)), ?2)'), lambda x: __is_real_const(x['?1']) and __is_real_const(x['?2'])),
    #Rule(parse_str('conj(sum(nsign(prod(j,?1)), ?2))'), ['?1', '?2'], parse_str('sum(prod(j,?1), ?2)'), lambda x: __is_real_const(x['?1']) and __is_real_const(x['?2'])),
    #Rule(parse_str('conj(?1)'), ['?1'], parse_str('?1'), lambda x: __is_real_const(x['?1'])),

    Rule(parse_str('prod(j, j)'), [], parse_str('-1')),
    Rule(parse_str('prod(0, ?1)'), ['?1'], parse_str('0')),
    Rule(parse_str('prod(?1, 0)'), ['?1'], parse_str('0')),
    Rule(parse_str('dagger(qgate(s, qb1))'), ['qb1'], parse_str('qgate(sdg, qb1)')),
    Rule(parse_str('dagger(qgate(sdg, qb1))'), ['qb1'], parse_str('qgate(s, qb1)')),
    Rule(parse_str('qgate(dagger(x), qb1)'), ['qb1'], parse_str('qgate(x, qb1)')),
    Rule(parse_str('square(x)'), [], parse_str('prod(x,x)')),

]
exp_misc_rules = [
    #13 This is specifically for gates with operator senario
    Rule(
        parse_str('exp( prod(t, N, M) )'),
        ['M', 'N', 't'],
        parse_str('prod( N, exp(prod(t, M)))'),
        lambda x : _is_const(x['t']) and x['N'].name == 'qgate'
    ),
    Rule(
        parse_str('exp( prod(M, N) )'),
        ['M', 'N'],
        parse_str('prod( exp(M), N)'),
        lambda x : x['N'].name == 'qgate'
    ), 

]

branching_rules_list = [
    Rule(parse_str('prod(w, sum(prod(x, y), prod(x, z)))'), ['w', 'x', 'y', 'z'], parse_str('prod(v, sum(prod(1, y), prod(1, z)))'), 
         lambda x: x['x'].name != '1', 
         {'v':(lambda x,_: ParsedNode('prod', [x['w'].copy(), x['x'].copy()]))}),
    Rule(parse_str('prod(w, sum(prod(x, y), prod(x2, z)))'), ['w', 'x', 'x2', 'y', 'z'], parse_str('prod(v, sum(prod(1, y), prod(-1, z)))'), 
         lambda x: x['x'].name != '1' and __check_equality_list(x, [(parse_str('x'), parse_str('nsign(x2)'))]), 
         {'v':(lambda x,_: ParsedNode('prod', [x['w'].copy(), x['x'].copy()]))}),
    Rule(parse_str('sum(prod(x, y), prod(x, z))'), ['x', 'y', 'z'], parse_str('prod(x, sum(prod(1, y), prod(1, z)))'), 
         lambda x: x['x'].name != '1'),
]

decomp_rules_list = [

    #distribute exp over sum
    Rule(parse_str('exp(sum(?1, ?2))'), ['?1', '?2'], parse_str('prod(exp(?1), exp(?2))'), 
        lambda x: x['?1'] != parse_str('0') and x['?2'] != parse_str('0') and (x['?1'].name != 'sum' or (len(x['?1'].children) == 2))
    ),

    #3
    Rule(parse_str('exp( t, comm(M, N)) )'),
      ['M', 'N', 't'],
      parse_str('BCH(prod(t2, sigma(3, new_i), N), prod(t2, sigma(3, new_i), M))'),
      lambda x : (
          __check_equality_list(x, [(parse_str('M'), parse_str('dagger(M)')), (parse_str('N'), parse_str('dagger(N)'))])
            and __is_squared_factor(x['t'], 1)
            ),
    {'t2':(lambda x,_ : __get_squared_factor(x['t'], 1) * 1j), 
     't3':(lambda x,_ : __get_squared_factor(x['t'], 1) * -1j),
     'new_i':(lambda _,env : __increment_counter(env.index_counters, 'qubit'))
    }),
    Rule(parse_str('exp( t, comm(M, N)) )'),
      ['M', 'N', 't'],
      parse_str('BCH(prod(t3, sigma(3, new_i), N), prod(t2, sigma(3, new_i), M))'),
      lambda x : (
          __check_equality_list(x, [(parse_str('M'), parse_str('dagger(M)')), (parse_str('N'), parse_str('dagger(N)'))])
            and __is_squared_factor(x['t'], -1)
            ),
    {'t2':(lambda x,_ : __get_squared_factor(x['t'], -1) * 1j),
     't3':(lambda x,_ : __get_squared_factor(x['t'], -1) * -1j),
     'new_i':(lambda _,env : __increment_counter(env.index_counters, 'qubit'))
    }),
    

    #4 
    Rule(parse_str('exp(prod(t, sigma(3, qb1), acomm(M, N) ))'),
      ['M', 'N', 't', 'qb1'],
      parse_str('BCH( prod(t2, sigma(1, qb1), M), prod(t2, sigma(2, qb1), N) )'),
      lambda x : (__check_equality_list(x, [(parse_str('M'), parse_str('dagger(M)')), (parse_str('N'), parse_str('dagger(N)'))]) and
            __is_squared_factor(x['t'], -1j)
      ),
      dict(t2=lambda x,_ : __get_squared_factor(x['t'], -1j) * 1j)),
    Rule(parse_str('exp(prod(t, sigma(2, qb1), acomm(M, N) ))'),
      ['M', 'N', 't', 'qb1'],
      parse_str('BCH( prod(t2, sigma(3, qb1), M), prod(t2, sigma(1, qb1), N) )'),
      lambda x : (__check_equality_list(x, [(parse_str('M'), parse_str('dagger(M)')), (parse_str('N'), parse_str('dagger(N)'))]) and
            __is_squared_factor(x['t'], -1j)
      ),
      dict(t2=lambda x,_ : __get_squared_factor(x['t'], -1j) * 1j)),
    Rule(parse_str('exp(prod(t, sigma(1, qb1), acomm(M, N) ))'),
      ['M', 'N', 't', 'qb1'],
      parse_str('BCH( prod(t2, sigma(2, qb1), M), prod(t2, sigma(3, qb1), N) )'),
      lambda x : (__check_equality_list(x, [(parse_str('M'), parse_str('dagger(M)')), (parse_str('N'), parse_str('dagger(N)'))]) and
            __is_squared_factor(x['t'], -1j)
      ),
      dict(t2=lambda x,_ : __get_squared_factor(x['t'], -1j) * 1j)),
    
    Rule(parse_str('exp(prod(t, sigma(3, qb1), acomm(M, N) ))'),
      ['M', 'N', 't', 'qb1'],
      parse_str('BCH( prod(nsign(t2), sigma(1, qb1), M), prod(t2, sigma(2, qb1), N) )'),
      lambda x : (__check_equality_list(x, [(parse_str('M'), parse_str('dagger(M)')), (parse_str('N'), parse_str('dagger(N)'))]) and
            __is_squared_factor(x['t'], 1j)
      ),
      dict(t2=lambda x,_ : __get_squared_factor(x['t'], 1j) * 1j)),
    Rule(parse_str('exp(prod(t, sigma(2, qb1), acomm(M, N) ))'),
      ['M', 'N', 't', 'qb1'],
      parse_str('BCH( prod(nsign(t2), sigma(3, qb1), M), prod(t2, sigma(1, qb1), N) )'),
      lambda x : (__check_equality_list(x, [(parse_str('M'), parse_str('dagger(M)')), (parse_str('N'), parse_str('dagger(N)'))]) and
            __is_squared_factor(x['t'], 1j)
      ),
      dict(t2=lambda x,_ : __get_squared_factor(x['t'], 1j) * 1j)),
    Rule(parse_str('exp(prod(t, sigma(1, qb1), acomm(M, N) ))'),
      ['M', 'N', 't', 'qb1'],
      parse_str('BCH( prod(nsign(t2), sigma(2, qb1), M), prod(t2, sigma(3, qb1), N) )'),
      lambda x : (__check_equality_list(x, [(parse_str('M'), parse_str('dagger(M)')), (parse_str('N'), parse_str('dagger(N)'))]) and
            __is_squared_factor(x['t'], 1j)
      ),
      dict(t2=lambda x,_ : __get_squared_factor(x['t'], 1j) * 1j)),

    #5
    Rule(parse_str('exp(prod(t, sigma(3, qb1), comm(M, N)))'),
      ['M', 'N', 't', 'qb1'],
      parse_str('BCH( prod(t2, N),  prod(t2, sigma(3, qb1), M))'),
      lambda x : ( __is_squared_factor(x['t'], -1j)),
      {'t2':(lambda x,_ : __get_squared_factor(x['t'], -1j) * 1j)}
      ),
    Rule(parse_str('exp(prod(t, sigma(3, qb1), comm(M, N)))'),
      ['M', 'N', 't', 'qb1'],
      parse_str('BCH( prod(nsign(t2), N),  prod(t2, sigma(3, qb1), M))'),
      lambda x : ( __is_squared_factor(x['t'], 1j)),
      {'t2':(lambda x,_ : __get_squared_factor(x['t'], 1j) * 1j)}
      ),

    #6
    Rule(parse_str('exp(prod(t, sigma(3, qb1), sum(prod(c1,M,N), prod(c2, Ndg, Mdg)))))'),
      ['M', 'N', 'Mdg', 'Ndg', 't', 'c1', 'c2', 'qb1'],
      parse_str('BCH( prod(t2, qgate(x, qb1), BOD(N, qb2), qgate(x, qb1)), prod(t2, BOD(M, qb3)))'),
      lambda x : x['M'].name != '1' and x['N'].name != '1' and __is_squared_factor(complex(x['t'].name)*complex(x['c1'].name), 1) 
        and __check_equality_list(x, [
          (parse_str('sum(prod(1,M,N), prod(-1,N,M))'), parse_str('0')),
          (parse_str('dagger(M)'), parse_str('Mdg')),
          (parse_str('dagger(N)'), parse_str('Ndg')),
          (parse_str('c1'), parse_str('nsign(c2)')),
        ]),
    dict(t2=lambda x,_ : __get_squared_factor(complex(x['t'].name)*complex(x['c1'].name), 1) * 1j,
         qb2=(lambda _,env : __increment_counter(env.index_counters, 'qubit')),
         qb3=(lambda _,env : __increment_counter(env.index_counters, 'qubit'))
         )),
    #6b
    Rule(parse_str('exp(prod(t, sigma(3, qb1), sum(prod(c1,M,N), prod(c2, Ndg, Mdg)))))'),
      ['M', 'N', 'Mdg', 'Ndg', 't', 'c1', 'c2', 'qb1'],
      parse_str('BCH( prod(nsign(t2), qgate(x, qb1), BOD(N, qb2), qgate(x, qb1)), prod(t2, BOD(M, qb3)))'),
      lambda x : x['M'].name != '1' and x['N'].name != '1' and __is_squared_factor(complex(x['t'].name)*complex(x['c1'].name), -1) 
        and __check_equality_list(x, [
          (parse_str('sum(prod(1,M,N), prod(-1,N,M))'), parse_str('0')),
          (parse_str('dagger(M)'), parse_str('Mdg')),
          (parse_str('dagger(N)'), parse_str('Ndg')),
          (parse_str('c1'), parse_str('nsign(c2)')),
        ]),
    dict(t2=lambda x,_ : __get_squared_factor(complex(x['t'].name)*complex(x['c1'].name), -1) * 1j,
         qb2=(lambda _,env : __increment_counter(env.index_counters, 'qubit')),
         qb3=(lambda _,env : __increment_counter(env.index_counters, 'qubit'))
         )),
    
    #7
    Rule(parse_str('exp(prod(t, sigma(3, qb1), sum(prod(c1,M,N), prod(c1, Ndg, Mdg)))))'),
      ['M', 'N', 'Mdg', 'Ndg', 't', 'c1', 'qb1'],
      parse_str('BCH( prod(t2, qgate(s, qb1), BOD(N, qb2), qgate(sdg, qb1)), prod(t2, qgate(x, qb1), BOD(M, qb3), qgate(x, qb1)))'),
      lambda x : x['M'].name != '1' and x['N'].name != '1' and __is_squared_factor(complex(x['t'].name)*complex(x['c1'].name), 1j) 
        and __check_equality_list(x, [
          (parse_str('sum(prod(1,M,N), prod(-1,N,M))'), parse_str('0')),
          (parse_str('dagger(M)'), parse_str('Mdg')),
          (parse_str('dagger(N)'), parse_str('Ndg')),
        ]),
    dict(t2=lambda x,_ : __get_squared_factor(complex(x['t'].name)*complex(x['c1'].name), 1j) * 1j,
         qb2=(lambda _,env : __increment_counter(env.index_counters, 'qubit')),
         qb3=(lambda _,env : __increment_counter(env.index_counters, 'qubit'))
         )),
    #7b
    Rule(parse_str('exp(prod(t, sigma(3, qb1), sum(prod(c1,M,N), prod(c1, Ndg, Mdg)))))'),
      ['M', 'N', 'Mdg', 'Ndg', 't', 'c1', 'qb1'],
      parse_str('BCH( prod(nsign(t2), qgate(s, qb1), BOD(N, qb2), qgate(sdg, qb1)), prod(t2, qgate(x, qb1), BOD(M, qb3), qgate(x, qb1)))'),
      lambda x : x['M'].name != '1' and x['N'].name != '1' and __is_squared_factor(complex(x['t'].name)*complex(x['c1'].name), -1j) 
        and __check_equality_list(x, [
          (parse_str('sum(prod(1,M,N), prod(-1,N,M))'), parse_str('0')),
          (parse_str('dagger(M)'), parse_str('Mdg')),
          (parse_str('dagger(N)'), parse_str('Ndg')),
        ]),
    dict(t2=lambda x,_ : __get_squared_factor(complex(x['t'].name)*complex(x['c1'].name), -1j) * 1j,
         qb2=(lambda _,env : __increment_counter(env.index_counters, 'qubit')),
         qb3=(lambda _,env : __increment_counter(env.index_counters, 'qubit'))
         )),
    
    #8
    Rule(parse_str('exp(prod(t, M, N))'),
      ['t', 'M', 'N'],
      parse_str('trotter(prod(t2, sigma(3, qb1), comm(M,N)), prod(t2, sigma(3, qb1), acomm(M,N)))'),
      lambda x : _is_const(x['t']) and __check_equality_list(x, [(parse_str('M'), parse_str('dagger(M)')), (parse_str('N'), parse_str('dagger(N)'))]),
      {'t2':(lambda x,_ : complex(x['t'].name) / 2),
       'qb1':(lambda _,env : __increment_counter(env.index_counters, 'qubit'))}
      ),

    #9
    Rule(parse_str('exp(prod(t, M, N))'),
      ['M', 'N', 't'],
      parse_str('BCH( prod(t2, qgate(s, qb1), BOD(M, qb2), qgate(sdg, qb1)), prod(t2, qgate(x, qb1), BOD(N, qb3), qgate(x, qb1)))'),
      lambda x : __is_squared_factor(x['t'], 1j) and x['M'].name != '1' and x['N'].name != '1' and __check_equality_list(x, [(parse_str('sum(prod(1,M,N), prod(-1,N,M))'), parse_str('0')), (parse_str('prod(1,M,N)'), parse_str('dagger(prod(1,M,N))')) ]),
      dict(t2=lambda x,_ : __get_squared_factor(x['t'], 2j) * 1j,
         qb1=(lambda _,env : __increment_counter(env.index_counters, 'qubit')),
         qb2=(lambda _,env : __increment_counter(env.index_counters, 'qubit')),
         qb3=(lambda _,env : __increment_counter(env.index_counters, 'qubit'))
         )),
    Rule(parse_str('exp(prod(t, M, N))'),
      ['M', 'N', 't'],
      parse_str('BCH( prod(nsign(t2), qgate(s, qb1), BOD(M, qb2), qgate(sdg, qb1)), prod(t2, qgate(x, qb1), BOD(N, qb3), qgate(x, qb1)))'),
      lambda x : __is_squared_factor(x['t'], -1j) and x['M'].name != '1' and x['N'].name != '1' and __check_equality_list(x, [(parse_str('sum(prod(1,M,N), prod(-1,N,M))'), parse_str('0')), (parse_str('prod(1,M,N)'), parse_str('dagger(prod(1,M,N))')) ]),
      dict(t2=lambda x,_ : __get_squared_factor(x['t'], -2j) * 1j,
         qb1=(lambda _,env : __increment_counter(env.index_counters, 'qubit')),
         qb2=(lambda _,env : __increment_counter(env.index_counters, 'qubit')),
         qb3=(lambda _,env : __increment_counter(env.index_counters, 'qubit'))
         )),

    #10
    Rule(parse_str('exp(prod(t, BOD(prod(M,N), qb1)))'),
      ['M', 'N', 't', 'qb1'],
      parse_str('prod(qgate(x, qb1), trotter(prod(t2, sigma(2, qb1), sum(prod(1,M,N),nsign(dagger(prod(1,M,N))))),  prod(t3, sigma(1, qb1), sum(prod(1,M,N),dagger(prod(1,M,N))))), qgate(x, qb1) )'),
      lambda x : _is_const(x['t']) and x['M'] != parse_str('1') and x['N'] != parse_str('1') and
        __check_equality_list(x, [(parse_str('sum(prod(1, M,N), prod(-1, N,M)))'), parse_str('0')) ]),
      dict(t2=lambda x,_ : complex(x['t'].name)/(2j), t3=lambda x,_ : complex(x['t'].name)/(2))
    ),

    #11
    Rule(parse_str('exp( prod(t, M, N)))'),
      ['M', 'N', 't'],
      parse_str('BCH( prod(?t, qgate(s, qb1), BOD(M, qb2), qgate(sdg, qb1) ),  prod(?t, qgate(x, qb1), BOD(N, qb3), qgate(x, qb1) ) )'),
      lambda x: (x['M'].name != '1' and x['N'].name != '1' and
        __check_equality_list(x, [(parse_str('prod(1,M,N)'), parse_str('prod(1,dagger(N),dagger(M))'))]) and
        _is_const(x['t'])
      ),
      {'?t':(lambda x,_: complex(x['t'].name)/2),
         "qb1":(lambda _,env : __increment_counter(env.index_counters, 'qubit')),
         "qb2":(lambda _,env : __increment_counter(env.index_counters, 'qubit')),
         "qb3":(lambda _,env : __increment_counter(env.index_counters, 'qubit'))
    }),
]

def __make_seq(pstring:str, qmode:ParsedNode, rev:bool = False):
    out = ParsedNode('prod', [])
    for i, c in enumerate(pstring.strip()):
        if rev:
            rotation = [parse_str(f"qgate(CR({math.pi}), {i}, {str(qmode)})")]
        else:
            rotation = [parse_str(f"qgate(CR(-{math.pi}), {i}, {str(qmode)})")]


        if c == 'I':
            continue
        elif c == 'X':
            rotation_full = [parse_str(f"qgate(h, {i})")] + rotation + [parse_str(f"qgate(h, {i})")]
        elif c == 'Y':
            rotation_full = [parse_str(f"qgate(h, {i})"), parse_str(f"qgate(sdg, {i})")] + rotation + [parse_str(f"qgate(s, {i})"), parse_str(f"qgate(h, {i})")]
        elif c == 'Z':
            rotation_full = rotation
        else:
            raise Exception("Improper pauli string, please adjust rule conditions")
        
        if rev:
            out.children = rotation_full + out.children
        else:
            out.children.extend(rotation_full)
    return out

def __get_j_coeff(pstring:str):
    return (-1j) ** len([True for c in pstring.strip() if c != 'I'])

sigma_z_rules = [
    Rule(parse_str('exp(prod(?1, sigma(1, qb1), ?2))'), ['?1', '?2', 'qb1'], parse_str('prod(qgate(h, qb1), exp(prod(?1, sigma(3, qb1), ?2)), qgate(h, qb1))')),
    Rule(parse_str('exp(prod(?1, sigma(1, qb1)))'), ['?1', 'qb1'], parse_str('prod(qgate(h, qb1), exp(prod(?1, sigma(3, qb1))), qgate(h, qb1))')),
    Rule(parse_str('exp(prod(sigma(1, qb1), ?2))'), ['?2', 'qb1'], parse_str('prod(qgate(h, qb1), exp(prod(sigma(3, qb1) ?2)), qgate(h, qb1),)')),
    Rule(parse_str('exp(prod(sigma(1, qb1)))'), ['qb1'], parse_str('prod(qgate(h, qb1), exp(prod( sigma(3, qb1) )), qgate(h, qb1),)')),

    Rule(parse_str('exp(prod(?1, sigma(2, qb1), ?2))'), ['?1', '?2', 'qb1'], parse_str('prod(qgate(h, qb1), qgate(sdg, qb1), exp(prod(?1, sigma(3, qb1), ?2)), qgate(s, qb1), qgate(h, qb1),)')),
    Rule(parse_str('exp(prod(?1, sigma(2, qb1)))'), ['?1', 'qb1'], parse_str('prod(qgate(h, qb1), qgate(sdg, qb1), exp(prod(?1, sigma(3, qb1))), qgate(s, qb1), qgate(h, qb1),)')),
    Rule(parse_str('exp(prod(sigma(2, qb1), ?2))'), ['?2', 'qb1'], parse_str('prod(qgate(h, qb1), qgate(sdg, qb1), exp(prod(sigma(3, qb1) ?2)), qgate(s, qb1), qgate(h, qb1),)')),
    Rule(parse_str('exp(prod(sigma(2, qb1)))'), ['qb1'], parse_str('prod(qgate(h, qb1), qgate(sdg, qb1), exp(prod( sigma(3, qb1) )), qgate(s, qb1), qgate(h, qb1),)')),

    Rule(
        parse_str('exp(prod(?t, paulistring(?ps), sum(prod(?alpha, dagger(?1)), prod(?alphac, ?1))))'),
        ['?t', '?1', '?ps', '?alpha', '?alphac'],
        parse_str("prod(?revseq, qgate(D(prod(?t, ?alpha, ?im)), ?1), ?seq)"),
        lambda x : __is_mode(x['?1']) and __check_equality_list(x, [
            (parse_str('prod(?t, ?alpha)'), parse_str('nsign(conj(prod(?t, ?alphac)))')),
        ]),
        {'?revseq':(lambda x,_:__make_seq(x['?ps'].name, x['?1'], rev=True)),
         '?seq':(lambda x,_:__make_seq(x['?ps'].name, x['?1'])),
         '?im':(lambda x,_:__get_j_coeff(x['?ps'].name))}
    ),
    Rule(
        parse_str('exp(prod(?t, paulistring(?ps), sum(prod(?alpha, ?1), prod(?alphac, dagger(?1)))))'),
        ['?t', '?1', '?ps', '?alpha', '?alphac'],
        parse_str("prod(?revseq, qgate(D(prod(?t, ?alpha, ?im)), ?1), ?seq)"),
        lambda x : __is_mode(x['?1']) and __check_equality_list(x, [
            (parse_str('prod(?t, ?alpha)'), parse_str('nsign(conj(prod(?t, ?alphac)))')),
        ]),
        {'?revseq':(lambda x,_:__make_seq(x['?ps'].name, x['?1'], rev=True)),
         '?seq':(lambda x,_:__make_seq(x['?ps'].name, x['?1'])),
         '?im':(lambda x,_:__get_j_coeff(x['?ps'].name))}
    ),
]

basic_gates_list = [

    #Eliminate sigma(0)
    Rule(parse_str('sigma(0, qb1)'), ['qb1'], parse_str('1')),

    #1
    #"""Pauli"""
    #2
    Rule(
        parse_str('exp(prod(?angle, sigma(3, qb1)))'),
        ['?angle', 'qb1'],
        parse_str('qgate(rz(?newangle), qb1)'),
        lambda x : _is_const(x['?angle']),
        {'?newangle':(lambda x,_: complex(x['?angle'].name) * -2 / 1j)}
    ),
    #3
    Rule(  
        parse_str('exp(prod(?angle, dagger(?1), ?1))'),
        ['?1', '?angle'],
        parse_str('qgate(R(?newangle), ?1)'),
        lambda x : 
            (__is_mode(x['?1']) and _is_const(x['?angle'])),
        {'?newangle':(lambda x,_: complex(x['?angle'].name)/ -1j)}
    ),
    #4
    Rule(
        parse_str('exp(prod(?t, sum(prod(?alpha, dagger(?1)), prod(?alphac, ?1))))'),
        ['?1', '?t', '?alpha', '?alphac'],
        parse_str('qgate(D(?newalpha), ?1)'),
        lambda x : 
            (__is_mode(x['?1']) and _is_const(x['?t']) and _is_const(x['?alpha']) and
             __check_equality_list(x, [
                 (parse_str('prod(?t, ?alpha)'), parse_str('nsign(conj(prod(?t, ?alphac)))')),
             ])
             ),
        {'?newalpha':(lambda x,_: complex(x['?t'].name) * complex(x['?alpha'].name))}
    ),

    #4b
    Rule(
        parse_str('exp(prod(?t, sum(prod(?alpha, ?1), prod(?alphac, dagger(?1)))))'),
        ['?1', '?t', '?alpha', '?alphac'],
        parse_str('qgate(D(?newalpha), ?1)'),
        lambda x : 
            (__is_mode(x['?1']) and _is_const(x['?t']) and _is_const(x['?alpha']) and
             __check_equality_list(x, [
                 (parse_str('prod(?t, ?alpha)'), parse_str('nsign(conj(prod(?t, ?alphac)))')),
             ])
             ),
        {'?newalpha':(lambda x,_: complex(x['?t'].name) * complex(x['?alpha'].name))}
    ),

    #5
    Rule(
        parse_str('exp(prod(?angle, sum(prod(?cf1, dagger(?1), ?2), prod(?cf1, ?1, dagger(?2)))))'),
        ['?1', '?2', '?angle', '?cf1'],
        parse_str('qgate(BS(?newangle, 0), ?1, ?2)'),
        lambda x: 
            _is_const(x['?cf1']) and
            (__is_mode(x['?1'])) and
            (__is_mode(x['?2'])) and
            (_is_const(x['?angle'])) and
            not __check_equality_list(x, [
            (parse_str("?1"), parse_str("?2")),
          ]),
        {'?newangle':(lambda x,_: complex(x['?cf1'].name) * complex(x['?angle'].name) * 2 / -1j)}
    ),
    Rule(
        parse_str('exp(prod(?angle, sum(prod(?cf1, dagger(?1), ?2), prod(?cf1, dagger(?2), ?1))))'),
        ['?1', '?2', '?angle', '?cf1'],
        parse_str('qgate(BS(?newangle, 0), ?1, ?2)'),
        lambda x: 
            _is_const(x['?cf1']) and
            (__is_mode(x['?1'])) and
            (__is_mode(x['?2'])) and
            (_is_const(x['?angle'])) and
            not __check_equality_list(x, [
            (parse_str("?1"), parse_str("?2")),
          ]),
        {'?newangle':(lambda x,_: complex(x['?cf1'].name) * complex(x['?angle'].name) * 2 / -1j)}
    ),
    Rule(
        parse_str('exp(prod(?angle, sum(prod(?cf1, ?1, dagger(?2)), prod(?cf1, dagger(?1), ?2))))'),
        ['?1', '?2', '?angle', '?cf1'],
        parse_str('qgate(BS(?newangle, 0), ?1, ?2)'),
        lambda x: 
            _is_const(x['?cf1']) and
            (__is_mode(x['?1'])) and
            (__is_mode(x['?2'])) and
            (_is_const(x['?angle'])) and
            not __check_equality_list(x, [
            (parse_str("?1"), parse_str("?2")),
          ]),
        {'?newangle':(lambda x,_: complex(x['?cf1'].name) * complex(x['?angle'].name) * 2 / -1j)}
    ),
    Rule(
        parse_str('exp(prod(?angle, sum(prod(?cf1, ?1, dagger(?2)), prod(?cf1, ?2, dagger(?1)))))'),
        ['?1', '?2', '?angle', '?cf1'],
        parse_str('qgate(BS(?newangle, 0), ?1, ?2)'),
        lambda x: 
            _is_const(x['?cf1']) and
            (__is_mode(x['?1'])) and
            (__is_mode(x['?2'])) and
            (_is_const(x['?angle'])) and
            not __check_equality_list(x, [
            (parse_str("?1"), parse_str("?2")),
          ]),
        {'?newangle':(lambda x,_: complex(x['?cf1'].name) * complex(x['?angle'].name) * 2 / -1j)}
    ),
    
    #6
    Rule(  
        parse_str('exp(prod(?angle, sigma(3, qb1), dagger(?1), ?1))'),
        ['?1', '?angle', 'qb1'],
        parse_str('qgate(CR(?newangle), qb1, ?1)'),
        lambda x : 
            (__is_mode(x['?1']) and _is_const(x['?angle'])),
        {'?newangle':(lambda x,_: complex(x['?angle'].name) * 2 / -1j)}
    ),
    
    #7
    #"""Seems same as 6"""
    #8
    Rule(
        parse_str('exp(prod(?t, sigma(3, qb1), sum(prod(?alpha, dagger(?1)), prod(?alphac, ?1))))'),
        ['?1', '?t', 'qb1', '?alpha', '?alphac'],
        parse_str('qgate(CD(?newalpha), qb1, ?1)'),
        lambda x : 
            (__is_mode(x['?1']) and _is_const(x['?t']) and _is_const(x['?alpha']) and
             __check_equality_list(x, [
                 (parse_str('prod(?t, ?alpha)'), parse_str('nsign(conj(prod(?t, ?alphac)))')),
             ])
             ),
        {'?newalpha':(lambda x,_: complex(x['?t'].name) * complex(x['?alpha'].name))}
    ),
    Rule(
        parse_str('exp(prod(?t, sum(prod(?alpha, sigma(3, qb1), dagger(?1)), prod(?alphac, sigma(3, qb1), ?1))))'),
        ['?1', '?t', 'qb1', '?alpha', '?alphac'],
        parse_str('qgate(CD(?newalpha), qb1, ?1)'),
        lambda x : 
            (__is_mode(x['?1']) and _is_const(x['?t']) and _is_const(x['?alpha']) and (__is_squared_factor(x['?t'], 1) or __is_squared_factor(x['?t'], -1)) and
             __check_equality_list(x, [
                 (parse_str('?alpha'), parse_str('nsign(conj(?alphac))')),
             ])
             ),
        {'?newalpha':(lambda x,_: complex(x['?t'].name) * complex(x['?alpha'].name))}
    ),

    #8b
    Rule(
        parse_str('exp(prod(?t, sigma(3, qb1), sum(prod(?alpha, ?1), prod(?alphac, dagger(?1)))))'),
        ['?1', '?t', 'qb1', '?alpha', '?alphac'],
        parse_str('qgate(CD(?newalpha), qb1, ?1)'),
        lambda x : 
            (__is_mode(x['?1']) and _is_const(x['?t']) and _is_const(x['?alpha']) and
             __check_equality_list(x, [
                 (parse_str('prod(?t, ?alpha)'), parse_str('nsign(conj(prod(?t, ?alphac)))')),
             ])
             ),
        {'?newalpha':(lambda x,_: complex(x['?t'].name) * complex(x['?alpha'].name))}
    ),
    Rule(
        parse_str('exp(prod(?t, sum(prod(?alpha, sigma(3, qb1), ?1), prod(?alphac, sigma(3, qb1), dagger(?1)))))'),
        ['?1', '?t', 'qb1', '?alpha', '?alphac'],
        parse_str('qgate(CD(?newalpha), qb1, ?1)'),
        lambda x : 
            (__is_mode(x['?1']) and _is_const(x['?t']) and _is_const(x['?alpha']) and (__is_squared_factor(x['?t'], 1) or __is_squared_factor(x['?t'], -1)) and
             __check_equality_list(x, [
                 (parse_str('?alpha'), parse_str('nsign(conj(?alphac))')),
             ])
             ),
        {'?newalpha':(lambda x,_: complex(x['?t'].name) * complex(x['?alpha'].name))}
    ),

    #9
    Rule(
        parse_str('exp(prod(?angle, sigma(3, qb1), sum(prod(?cf1, dagger(?1), ?2), prod(?cf1, ?1, dagger(?2)))))'),
        ['?1', '?2', '?angle', '?cf1', 'qb1'],
        parse_str('qgate(CBS(?newangle, 0), qb1, ?1, ?2)'),
        lambda x: 
            _is_const(x['?cf1']) and
            (__is_mode(x['?1'])) and
            (__is_mode(x['?2'])) and
            (_is_const(x['?angle'])) and
            not __check_equality_list(x, [
            (parse_str("?1"), parse_str("?2")),
          ]),
        {'?newangle':(lambda x,_: complex(x['?cf1'].name) * complex(x['?angle'].name) * 2 / -1j)}
    ),
    Rule(
        parse_str('exp(prod(?angle, sigma(3, qb1), sum(prod(?cf1, dagger(?1), ?2), prod(?cf1, dagger(?2), ?1))))'),
        ['?1', '?2', '?angle', '?cf1', 'qb1'],
        parse_str('qgate(CBS(?newangle, 0), qb1, ?1, ?2)'),
        lambda x: 
            _is_const(x['?cf1']) and
            (__is_mode(x['?1'])) and
            (__is_mode(x['?2'])) and
            (_is_const(x['?angle'])) and
            not __check_equality_list(x, [
            (parse_str("?1"), parse_str("?2")),
          ]),
        {'?newangle':(lambda x,_: complex(x['?cf1'].name) * complex(x['?angle'].name) * 2 / -1j)}
    ),
    Rule(
        parse_str('exp(prod(?angle, sigma(3, qb1), sum(prod(?cf1, ?1, dagger(?2)), prod(?cf1, dagger(?1), ?2))))'),
        ['?1', '?2', '?angle', '?cf1', 'qb1'],
        parse_str('qgate(CBS(?newangle, 0), qb1, ?1, ?2)'),
        lambda x: 
            _is_const(x['?cf1']) and
            (__is_mode(x['?1'])) and
            (__is_mode(x['?2'])) and
            (_is_const(x['?angle'])) and
            not __check_equality_list(x, [
            (parse_str("?1"), parse_str("?2")),
          ]),
        {'?newangle':(lambda x,_: complex(x['?cf1'].name) * complex(x['?angle'].name) * 2 / -1j)}
    ),
    Rule(
        parse_str('exp(prod(?angle, sigma(3, qb1), sum(prod(?cf1, ?1, dagger(?2)), prod(?cf1, ?2, dagger(?1)))))'),
        ['?1', '?2', '?angle', '?cf1', 'qb1'],
        parse_str('qgate(CBS(?newangle, 0), qb1, ?1, ?2)'),
        lambda x: 
            _is_const(x['?cf1']) and
            (__is_mode(x['?1'])) and
            (__is_mode(x['?2'])) and
            (_is_const(x['?angle'])) and
            not __check_equality_list(x, [
            (parse_str("?1"), parse_str("?2")),
          ]),
        {'?newangle':(lambda x,_: complex(x['?cf1'].name) * complex(x['?angle'].name) * 2 / -1j)}
    ),
    
    #10
    Rule(
        parse_str('prod(?t, sigma(1, qb1), sum(prod(?angle, dagger(?1)), prod(?anglec, ?1)))'),
        ['?1', '?angle', '?anglec', 'qb1', '?t'],
        parse_str('qgate(RB(?newangle), qb1, ?1)'),
        lambda x : 
            (__is_mode(x['?1'])) and
            (__is_squared_factor(x['?t'], 1) or __is_squared_factor(x['?t'], -1)) and
            __check_equality_list(x, [
            (parse_str("?angle"), parse_str("conj(?anglec)")),
          ]),
        {'?newangle':(lambda x,_: complex(x['?t'].name) * complex(x['?angle'].name) -1j)}
    ),
    
    #Some decomp gates have been moved here

    #1
    Rule(parse_str('trotter(prod(t1, M), prod(t2, N))'),
     ['M', 'N', 't1', 't2'],
     parse_str('prod(exp(prod(t1, M)), exp(prod(t2, N)))'),
     lambda x : _is_const(x['t1']) and _is_const(x['t2'])),

    #2
    Rule(parse_str('BCH(prod(t1, M), prod(t2, N))'),
      ['M', 'N', 't1', 't2'],
      parse_str('prod(exp(prod(t1, M)), exp(prod(t2, N)), exp(prod(nsign(t1), M)), exp(prod(nsign(t2), N)))'),
      lambda x : _is_const(x['t1']) and _is_const(x['t2'])
    ),

    #12
    Rule(parse_str('exp( prod(?a, BOD(M, qb1)) )'),
        ['?a','M', 'qb1'],
        #exp(𝑖(𝜋/2)𝑎†𝑎)exp(𝑖(𝛼 (𝑎† + 𝑎)) ⊗ 𝜎𝑦)exp(−𝑖(𝜋/2)𝑎†𝑎)exp(𝑖(𝛼 (𝑎† + 𝑎)) ⊗ 𝜎𝑥 )
        #R(-pi/2)h()sdg()CD(j alpha)s()h()R(pi/2)h()CD(j alpha)h()
        parse_str(f'prod(qgate(R({-math.pi/2}), M), qgate(h, qb1), qgate(sdg, qb1), qgate(CD(?alphai), qb1, M), qgate(s, qb1), qgate(h, qb1), qgate(R({math.pi/2}), M), qgate(h, qb1), qgate(CD(?alphai), qb1, M), qgate(h, qb1))'),
        #parse_str('prod( qgate(h), qgate(CD(-1j)), qgate(h), qgate(R(pi/2)), qgate(h), qgate(sdg), qgate(CD(j)), qgate(s), qgate(h),  qgate(R(-pi/2)))'),
        lambda x: (__is_mode(x['M']) and
            _is_const(x['?a'])
            and (complex(x['?a'].name)/ (2j) == complex.conjugate(complex(x['?a'].name)/ (2j)))
        ),
        {'?alpha':(lambda x,_: complex(x['?a'].name)/ (2j)), '?alphai':(lambda x,_: complex(x['?a'].name)/ (2))}),
    #13
    Rule(parse_str('exp( prod(?a, BOD(dagger(M), qb1)) )'),
        ['?a','M', 'qb1'],
        #parse_str( 'prod( qgate(h), qgate(CD(i)), qgate(h), qgate(R(pi/2)), qgate(h), qgate(sdg), qgate(CD(i)), qgate(s), qgate(h),  qgate(R(-pi/2)))'  ),
        parse_str(f'prod(qgate(R({-math.pi/2}), M), qgate(h, qb1), qgate(sdg, qb1), qgate(CD(?alphai), qb1, M), qgate(s, qb1), qgate(h, qb1), qgate(R({math.pi/2}), qb1, M), qgate(h, qb1), qgate(CD(nsign(?alphai)), qb1, M), qgate(h, qb1))'),
        lambda x: (__is_mode(x['M']) and
            _is_const(x['?a'])
            and (complex(x['?a'].name)/ (2j) == complex.conjugate(complex(x['?a'].name)/ (2j)))
        ),
        {'?alpha':(lambda x,_: complex(x['?a'].name)/ (2j)), '?alphai':(lambda x,_: complex(x['?a'].name)/ (2))}),
]

#Example of output str: CR(-2) q[1], qm[1]
def to_cvdv_str(qgate:ParsedNode):
    gate_name = qgate.children[0].name
    if len(qgate.children[0].children) > 0:
        param_list = []
        for param in qgate.children[0].children:
            p = str(param.name)
            p = p.strip("() ")
            #p = p.replace('j', 'i')
            param_list.append(p)
        gate_name += f"({', '.join(param_list)})"

    qm_list = []
    q_list = []
    for q in qgate.children[1:]:
        if q.name == 'b' and len(q.children) == 1:
            qm_list.append(f"qm[{q.children[0].name}]")
        else:
            q_list.append(f"q[{q.name}]")
    out_list = q_list + qm_list

    out = gate_name
    if len(out_list) > 0:
        out += " " + ", ".join(out_list)
    return out


def main(input_string:str, qbcount=100, debug=False, write_ruleslist:str|None = None):
    node = parse_str(input_string)

    simplify_ops(node)
    if debug:
        print(node)

    if debug:
        print()
    greedy_rules_list = basic_gates_list + commute_rules_list + exp_misc_rules + basic_rules_list + gate_cancel_rules
    branching_rules = branching_rules_list
    recursive_rules = decomp_rules_list + sigma_z_rules

    if write_ruleslist != None:
        full_list = greedy_rules_list + branching_rules + recursive_rules
        with open(write_ruleslist, 'w') as fp:
            for r in sorted(full_list, key=lambda x : x.id):
                fp.write(f"{r.id}: {r}\n")

    #start_time = timer()

    def is_terminal_node(x:ParsedNode):
        z = x
        simplify_ops(z)
        return all([i.name in ('gate', 'qgate') for i in z.children]) or z.name in ('gate', 'qgate')
    
    init_env = StateEnv()
    init_env.index_counters['qubit'] = qbcount

    final, out_res, stats = apply_rules_list_full_search(node, recursive_rules, branching_rules, greedy_rules_list, end_condition=is_terminal_node, dead_end_patterns=terminate_patterns, env=init_env)
    #out_res, stats = apply_rules_list_full_search(node, full_list, greedy_rules_list, dead_end_patterns=terminate_patterns)
    
    '''print("~~ output ~~")
    for env, out in sorted(out_res, key=lambda x : len([True for y in x[1].children if y.name in ('gate', 'qgate')])):
        steps = env.history
        for rule, ir in steps:
            print(f"{'>' * env.layer} {rule.id}: {ir}")
        print([r for r,_ in steps])
        print(Counter([r for r,_ in steps]))
        print("end_seq:", out)
        print()

    print(f'end time: {timer() - start_time}')
    print()
    print(stats)
    print()'''

    if final != None:

        result = final[1]
        if result.name in ('gate', 'qgate'):
            if debug:
                print(to_cvdv_str(result))
            return final[0], [to_cvdv_str(result)], stats
        else:
            for c in result.children:
                if debug:
                    print(to_cvdv_str(c))
            return final[0], [to_cvdv_str(c) for c in result.children], stats
    else:
        raise Exception(f"Line {node} does not decompose")

    
    
if __name__ == '__main__':
    qbcount = 100
    if len(sys.argv) >= 3:
        try:
            qbcount = int(sys.argv[2])
        except ValueError:
            pass
    debug = None
    if '--debug' in sys.argv:
        debug = True
    with open(sys.argv[1]) as fp:
        _, res, _ = main(fp.read(), qbcount, debug)
        for s in res:
            print(s+";")
