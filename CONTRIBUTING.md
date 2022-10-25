<div id="top"></div>

# Contribuindo com o MOGESTpy

<!-- TABLE OF CONTENTS -->

Queremos tornar a contribuição para este projeto o mais fácil e transparente possível, seja:

- Relatando um bug;
- Discutir o estado atual do código;
- Enviando uma correção;
- Proposta de novos recursos;
- Tornando-se um mantenedor.

## Escreva _Commits_ de Forma Consistente

Usamos o seguinte conjunto de convenções sobre como escrever _commits_ (baseado em [sugestões de boas práticas da comunidade de desenvolvimento](https://ruanbrandao.com.br/2020/02/04/como-fazer-um-bom-commit)):

### Apenas uma mudança por _commit_

Um _commit_ existe para fazer uma coisa, ou seja, é importante o _commit_ fazer apenas uma mudança. Isso ajuda a entender melhor como o projeto evoluiu ao longo do tempo, além de fazer com que seja mais fácil reverter o _commit_ caso algo dê errado;

### Idioma do _Commit_

Utilizamos o idioma Português brasileiro (PT-BR).

### Título do _Commit_

- O **título do _commit_** consiste de uma frase sucinta que diz **o que o _commit_ faz**. Ele deve ser sucinto, descritivo e específico. Ter um título específico ajuda a diferenciar _commit_s similares pela mensagem.
- Começar o título da mensagem do _commit_ com letra maiúscula;
- Não utilizar ponto final no título do _commit_;
- Utilizar uma linha em branco para separar título e corpo do _commit_;
- Limitar a linha do título do _commit_ a 72 caracteres;
- Utilizar o tempo verbal correto para o título da mensagem: 
    - Ao escrever a mensagem do _commit_ em **Português brasileiro (PT-BR)** conjugue o verbo no **presente do indicativo** (refato**ra**, atuali**za**, remo**ve**, etc.), utilizando a **terceira pessoa do singular** (ele/ela);
    - Ex.: 
        - `Refatora sistema X para melhorar legibilidade` ✅
        - `Atualiza documentação de instalação do projeto` ✅
        - `Remove métodos obsoletos` ✅
        - `Aplicando atualizações de pacotes` ❌
        - `Atualizar links do README.md` ❌

### Mensagem de um _Commit_

- A **mensagem de um _commit_** pode ter uma descrição longa, ou corpo da mensagem, e deve focar em dizer **o que foi feito** no _commit_ **e por que** essa mudança aconteceu.
- Limitar as linhas do corpo do _commit_ a 72 caracteres;

### Testes Automatizados

- Os testes automatizados do projeto devem passar sem erros antes de fazer um _commit_.

<p align="right">(<a href="#top">voltar ao topo</a>)</p>

## Desenvolvemos com Github

Usamos o github para hospedar o código, rastrear _issues_ e solicitações de recursos, bem como aceitar _pull requests_.

### Usamos o [GitHub Flow](https://guides.github.com/introduction/flow/index.html)

Todas as alterações de código acontecem por meio de _pull requests_. As _pull requests_ são a melhor maneira de propor alterações na base de código (usamos [GitHub Flow](https://guides.github.com/introduction/flow/index.html)):

1. Faça um clone do repositório e crie sua _branch_ a partir de `main` (_patches_) ou `development` (_features_ em novas _releases_);
2. Se você adicionou um código que deve ser testado, adicione testes;
3. Se você alterou algum componente, atualize a documentação;
4. Certifique-se de que o conjunto de testes passe;
5. Certifique-se de que seu código esteja de acordo com o estilo do projeto;
6. Crie sua _pull requests_!

<p align="right">(<a href="#top">voltar ao topo</a>)</p>

### Reportar Bugs Usando os [_Issues_](https://github.com/LabSid-USP/SSD2Petrobras/issues) do Github 

Usamos _issues_ do GitHub para rastrear bugs. Relate um bug [abrindo um novo _issue_](https://github.com/LabSid-USP/SSD2Petrobras/issues/new/choose).

<p align="right">(<a href="#top">voltar ao topo</a>)</p>

## Use um Estilo de Codificação Consistente

Utilizamos o guia de estilos da *Python Enhancement Proposals* [PEP 8](https://peps.python.org/pep-0008/)

Algumas outras diretrizes sobre a codificação são:

**Idioma no código**

Para manter a coerência dos códigos, recomenda-se:

- Nomes de funções e classes em inglês EN-US;
- Documentação dentro do código em português PT-BR;
- Comentários em português PT-BR.

```python
def do_something(parameter): # Comentário
    """
    Sumário da função. Eg: Essa função é fera

    param parameter: parametro de entrada
    type parameter: str
    param return: saída da função
    type return: str
    """

    variable_name = parameter + '_foo'

    return variable_name
```

<p align="right">(<a href="#top">voltar ao topo</a>)</p>