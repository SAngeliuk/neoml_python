# ����� CGlobalMeanPoolingLayer

<!-- TOC -->

- [����� CGlobalMeanPoolingLayer](#�����-cglobalmeanpoolinglayer)
    - [��������� ���������](#���������-���������)
    - [�����](#�����)
    - [������](#������)

<!-- /TOC -->

����� ��������� ����, ����������� �������� `Mean Pooling` ��� ������������� `Height`, `Width`, `Depth`.

## ��������� ���������

���� �� ����� ��������� ����������.

## �����

�� ������������ ���� �������� [����](../DnnBlob.md) � ������� �����������:

- `BatchLength * BatchWidth * ListSize` - ���������� ����������� � ������;
- `Height` - ������ �����������;
- `Width` - ������ �����������;
- `Depth` - ������� �����������;
- `Channels` - ���������� ������� � �����������.

## ������

������������ ����� �������� ���� �������:

- `BatchLength` ����� `BatchLength` �����;
- `BatchWidth` ����� `BatchWidth` �����;
- `ListSize` ����� `ListSize` �����;
- `Height` ����� `1`;
- `Width` ����� `1`;
- `Depth` ����� `1`;
- `Channels` ����� `Channels` �����.
