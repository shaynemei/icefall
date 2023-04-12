import logging
from pathlib import Path
import k2


logging.basicConfig(
    format = "%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
    level = 10
)

def generate_context_graph_nfa(
    words_pieces_list: list,
    backoff_id: int,
    bonus_per_token: float = 0.1,
    vocab_size = 5,
):
    flip = -1

    fsa_list = []
    fsa_sizes = []
    for words_pieces in words_pieces_list:
        start_state = 0
        # if the path go through this state, then a word boundary is detected
        boundary_state = 1
        next_state = 2  # the next un-allocated state, will be incremented as we go.
        arcs = []

        # arcs.append([start_state, start_state, backoff_id, 0, 0.0])
        for token_id in range(vocab_size):
            arcs.append([start_state, start_state, token_id, 0, 0.0])
            arcs.append([boundary_state, start_state, token_id, 0, 0.0])  # TODO: Adding this line here degrades performance. Why?

        for tokens in words_pieces:
            assert len(tokens) > 0
            cur_state = start_state
            
            # static/constant bonus per token
            # my_bonus_per_token = flip * bonus_per_token * biasing_list[word]
            # my_bonus_per_token = flip * 1.0 / len(tokens) * biasing_list[word]
            my_bonus_per_token = flip * bonus_per_token  # TODO: support weighted biasing list

            for i in range(len(tokens) - 1):
                arcs.append(
                    [cur_state, next_state, tokens[i], 0, my_bonus_per_token]
                )
                if i == 0:
                    arcs.append(
                        [boundary_state, next_state, tokens[i], 0, my_bonus_per_token]
                    )

                cur_state = next_state
                next_state += 1

            # now for the last token of this word
            i = len(tokens) - 1
            arcs.append(
                [cur_state, boundary_state, tokens[i], 0, my_bonus_per_token]
            )
        
        # for token_id in range(sp.vocab_size()):
        #     token = sp.id_to_piece(token_id)
        #     if token.startswith("▁"):
        #         arcs.append([boundary_state, start_state, token_id, 0, 0.0])

        final_state = next_state
        arcs.append([start_state, final_state, -1, -1, 0])
        arcs.append([boundary_state, final_state, -1, -1, 0])
        arcs.append([final_state])

        arcs = sorted(arcs, key=lambda arc: arc[0])
        arcs = [[str(i) for i in arc] for arc in arcs]
        arcs = [" ".join(arc) for arc in arcs]
        arcs = "\n".join(arcs)

        fsa = k2.Fsa.from_str(arcs, acceptor=False)
        fsa = k2.arc_sort(fsa)
        # fsa = k2.determinize(fsa)  # No weight pushing is needed.
        fsa_sizes.append((fsa.shape[0], fsa.num_arcs))  # (n_states, n_arcs)

        # fsa = k2_to_openfst(fsa, olabels="aux_labels")
        fsa_list.append(fsa)
    
    return fsa_list, fsa_sizes


def main():

    words_pieces_list = [[
        [1, 2, 2],
        [1, 2, 1],
    ]]

    fsa_list, fsa_sizes = generate_context_graph_nfa(
        words_pieces_list=words_pieces_list,
        backoff_id=500,
        bonus_per_token=0.1,
        vocab_size=3,
    )

    fsa = fsa_list[0]
    fsa.draw('simple_fsa.svg')


if __name__ == '__main__':
    main()


