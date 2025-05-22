from src.pattern_match import *
from src.pattern_match_rules import *


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
