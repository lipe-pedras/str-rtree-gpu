CONSTRUÇÃO DE ÍNDICES ESPACIAIS R-TREE EM GPU PARA CONJUNTOS DE DADOS MAIORES QUE A MEMÓRIA DISPONÍVEL

Nome do(a) estudante (IC), Nome do(a) orientador(a)

1 Instituto de Matemática e Computação, Universidade Federal de Itajubá

RESUMO

Índices espaciais R-Tree são essenciais para consultas geográficas, mas custosos de construir em grandes volumes. A literatura de construção em GPU assume que os dados cabem na memória do dispositivo — hipótese frágil, pois essa é a memória mais cara e escassa da máquina. Implementou-se o algoritmo Sort-Tile-Recursive em GPU para conjuntos maiores que ambas as memórias. Para 300 milhões de retângulos (7,2 GB), o índice foi construído e validado em 40,7 s, com ordenação 18 vezes mais rápida que na CPU, embora o ganho total fosse de 1,28 vez.

Palavras-chave: Índices espaciais. R-Tree. GPU. Processamento fora do núcleo. Bancos de dados.

Eixo temático: [preencher]

Área de Conhecimento: Ciência da Computação

Campus: Itajubá

INTRODUÇÃO

O volume de dados espaciais cresce mais rápido que a memória disponível para processá-los. A R-Tree é a estrutura de indexação mais usada para consultas por região, mas construí-la sobre bilhões de objetos é caro. A construção em lote (bulk loading) resolve isso ordenando os dados e empacotando as folhas diretamente, o que torna o problema dominado por ordenação — operação em que a GPU é notoriamente eficiente.

Toda a literatura consultada, porém, pressupõe que o conjunto de dados já resida na memória global da GPU. Essa hipótese é explicitada em Luo et al. (2012) e os experimentos das demais obras usam de 10⁵ a 10⁶ objetos. Como a memória de vídeo é o recurso mais caro e de menor capacidade do computador, tal pressuposto exclui justamente os casos de interesse prático. O objetivo deste trabalho é responder: dado que os dados não cabem na GPU, qual é a melhor forma de utilizá-la?

REFERENCIAL TEÓRICO

Uma R-Tree (GUTTMAN, 1984) organiza objetos em retângulos envolventes mínimos (MBR) hierárquicos, permitindo descartar subárvores que não intersectam a consulta. O algoritmo Sort-Tile-Recursive (Leutenegger; Edgington; Lopez, 1997) constrói árvores compactas ordenando os objetos pelo eixo X, dividindo-os em faixas verticais, reordenando cada faixa pelo eixo Y e agrupando os objetos em folhas — repetindo o processo sobre os nós gerados até restar a raiz.

A decisão de usar GPU costuma basear-se na intensidade operacional — operações por byte transferido (WILLIAMS; WATERMAN; PATTERSON, 2009). Esse critério é desfavorável aqui: ordenar exige cerca de uma comparação por elemento por passagem, valor centenas de vezes inferior ao ponto de equilíbrio da placa utilizada. A GPU ainda vence, porém, porque sua largura de banda de memória (192 GB/s) supera a da CPU (76,8 GB/s) e porque o trabalho é massivamente paralelo.

METODOLOGIA

Implementou-se em CUDA um construtor externo em quatro fases: ordenação por X em blocos na GPU; intercalação k-vias das partições ordenadas; ordenação por Y de cada faixa com empacotamento das folhas; e construção dos níveis internos. Cada nó ocupa exatamente uma página de 4096 bytes alinhada, contendo até 170 pares (MBR, identificador) dos filhos.

Três mecanismos viabilizam o caso fora do núcleo: partições ordenadas são mantidas em RAM enquanto houver orçamento e apenas o excedente vai ao disco; a intercalação é feita em passagem única com árvore de perdedores, custo mínimo em operações de E/S; e a ordenação sobrepõe-se à E/S por duplo buffer, com uma única linha de execução de disco, pois dois fluxos simultâneos degradam o SSD.

Para isolar o ganho da GPU, desenvolveu-se uma versão idêntica em CPU, alterando somente a ordenação, executável com uma ou várias linhas de execução. Os testes usaram um notebook com GPU RTX 3050 (4 GB), processador Ryzen 5 6600H (6 núcleos) e 14 GB de RAM, em modo de desempenho. A corretude foi verificada percorrendo-se toda a árvore: cada objeto deve aparecer exatamente uma vez e todo MBR deve conter sua subárvore.

RESULTADOS

Para 300 milhões de retângulos (7,2 GB, contra orçamento de 5 GB de RAM e 2,9 GB de memória de vídeo), o índice foi construído em 40,7 s, gerando 1.775.150 páginas e altura 4 — valores idênticos aos previstos analiticamente. A verificação confirmou 300.000.000 de 300.000.000 objetos indexados exatamente uma vez, com ocupação de 100% das folhas.

A ordenação na GPU foi 18 vezes mais rápida que na CPU com todos os núcleos, mas o ganho total foi de apenas 1,28 vez (39,7 s contra 50,8 s). A explicação está na Lei de Amdahl: a ordenação representa 42,7% do tempo da versão em CPU e apenas 3,0% da versão em GPU. O restante é movimentação de dados — leitura de disco (23,3 s) e intercalação (11,7 s).

A substituição do heap binário por árvore de perdedores com particionamento paralelo tornou a intercalação em memória 12,1 vezes mais rápida. Constatou-se ainda que a alocação do vetor de saída custava mais que a própria intercalação, por inicializar 1,4 GB desnecessariamente.

Nas consultas por região, cada página lida produziu de 169,5 a 170,0 decisões de poda, contra o máximo teórico de 170, e o índice respondeu 8.386 vezes mais rápido que a varredura completa, com resultados idênticos.

CONSIDERAÇÕES FINAIS

Demonstrou-se ser viável indexar, em uma GPU de 4 GB, conjuntos que não cabem nem na memória de vídeo nem na memória principal. O ganho proporcionado pela GPU, contudo, é limitado pela fração do processo que de fato é ordenação, e essa fração diminui à medida que os dados crescem. Construção de índices fora do núcleo é, essencialmente, um problema de movimentação de dados. Os trabalhos futuros concentram-se nesse ponto: antecipar as leituras da intercalação e reaproveitar buffers entre as fases.

REFERÊNCIAS

GUTTMAN, Antonin. R-trees: a dynamic index structure for spatial searching. In: ACM SIGMOD INTERNATIONAL CONFERENCE ON MANAGEMENT OF DATA, 1984, Boston. Proceedings [...]. New York: ACM, 1984. p. 47-57.

LEUTENEGGER, Scott T.; EDGINGTON, Jeffrey M.; LOPEZ, Mario A. STR: a simple and efficient algorithm for R-tree packing. Hampton: ICASE, 1997. (ICASE Report No. 97-14; NASA Contractor Report 201661).

LUO, Lijuan; WONG, Martin D. F.; LEONG, Lance. Parallel implementation of R-trees on the GPU. In: ASIA AND SOUTH PACIFIC DESIGN AUTOMATION CONFERENCE, 17., 2012, Sydney. Proceedings [...]. Piscataway: IEEE, 2012. p. 353-358.

WILLIAMS, Samuel; WATERMAN, Andrew; PATTERSON, David. Roofline: an insightful visual performance model for multicore architectures. Communications of the ACM, New York, v. 52, n. 4, p. 65-76, 2009.

AGRADECIMENTOS

À Universidade Federal de Itajubá pelo apoio institucional a esta Iniciação Científica.
